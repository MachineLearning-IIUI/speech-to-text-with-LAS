import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F
import numpy as np
from vocab import LABEL_MAP

from vocab import LABEL_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Listener(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers):
        super(Listener, self).__init__()
        self.input_size = input_size
        self.nlayers = nlayers
        self.lstm_list = []
        for i in range(nlayers):
            if i == 0:
                lstm = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               bidirectional=True)
            else:
                lstm = nn.LSTM(input_size=hidden_size * 2 * 2,
                               hidden_size=hidden_size,
                               num_layers=1,
                               bidirectional=True)

            self.lstm_list.append(lstm)
        self.lstm_list = nn.ModuleList(self.lstm_list)

    def forward(self, inputs_list): # batch_size * var_seq_len * 40
        batch_size = len(inputs_list)

        inputs_length = [len(utterance) for utterance in inputs_list] # original utterance lengths
        max_length = max(inputs_length)
        inputs_length = torch.LongTensor(inputs_length)
        outputs_length = inputs_length // 8 # output utterance lengths

        # longest_len * batch_size * 40
        padded_inputs = rnn.pad_sequence(inputs_list).to(DEVICE)

        for i in range(self.nlayers):
            # lstm_output: longest_len * batch_size * (hidden_size*2)
            lstm_outputs, _ = self.lstm_list[i](padded_inputs)

            if i != self.nlayers - 1:
                longest_len = lstm_outputs.shape[0]
                dim = lstm_outputs.shape[2]
                # chop off the extra
                if longest_len % 2 != 0:
                    lstm_outputs = lstm_outputs[:longest_len-1]
                longest_len = longest_len // 2
                dim = dim * 2
                # transpose lstm output to batch_size * longest_len * dim
                padded_inputs = lstm_outputs.transpose(0, 1)
                # reshape to batch_size * (longest_len/2) * (dim*2)
                padded_inputs = padded_inputs.reshape(batch_size, longest_len, dim)
                # transpose back to (longest_len/2) * batch_size * (dim*2)
                padded_inputs = padded_inputs.transpose(0, 1)

        # batch_size * longest_len * (hidden_size*2)
        lstm_outputs = lstm_outputs.transpose(0, 1)

        return lstm_outputs, outputs_length

class Speller(nn.Module):
    def __init__(self, listener_hidden_dim, speller_hidden_dim,
                 embedding_dim, class_size, key_dim, value_dim, batch_size):
        super(Speller, self).__init__()
        rnn_input_size = embedding_dim + value_dim
        self.rnn_layer1 = nn.LSTMCell(input_size=rnn_input_size, hidden_size=speller_hidden_dim)
        self.rnn_layer2 = nn.LSTMCell(input_size=speller_hidden_dim, hidden_size=speller_hidden_dim)
        self.attention = AttentionContext(speller_hidden_dim, listener_hidden_dim, key_dim, value_dim)
        self.embed = nn.Embedding(num_embeddings=class_size, embedding_dim=embedding_dim)

        linear_in_features = speller_hidden_dim + value_dim
        self.char_distribution_linear = nn.Linear(in_features=linear_in_features, out_features=class_size)
        self.softmax = nn.Softmax(dim=-1)

        rnn1_hidden_state = nn.Parameter(torch.randn(batch_size, speller_hidden_dim)).to(DEVICE)
        rnn1_cell_state = nn.Parameter(torch.randn(batch_size, speller_hidden_dim)).to(DEVICE)
        rnn2_hidden_state = nn.Parameter(torch.randn(batch_size, speller_hidden_dim)).to(DEVICE)
        rnn2_cell_state = nn.Parameter(torch.randn(batch_size, speller_hidden_dim)).to(DEVICE)
        self.rnn1_hc = (rnn1_hidden_state, rnn1_cell_state)
        self.rnn2_hc = (rnn2_hidden_state, rnn2_cell_state)

    def forward(self, listener_output, outputs_length, targets, teacher_forcing):
        timestep = max([len(seq) for seq in targets]) - 1
        targets_length_for_loss = [len(label)-1 for label in targets] # original label lengths
        # batch_size * max_transcript_len (LongTensor)
        padded_targets = rnn.pad_sequence(targets, batch_first=True).long().to(DEVICE)
        targets_for_loss = rnn.pad_sequence(targets, batch_first=True, padding_value=-1).long().to(DEVICE)
        targets_for_loss = targets_for_loss[:,1:] # only need targets starting from index 1

        probs = []
        predictions = []
        attentions = []

        for i in range(timestep):
            if i != 0:
                if np.random.random() < teacher_forcing:
                    # embed input is a 1d tensor
                    # batch_size * embedding_dim
                    embed = self.embed(padded_targets[:,i-1])
                else:
                    embed = self.embed(preds)
                inputs = torch.cat((embed, context), dim=1)
                rnn1_hc = self.rnn_layer1(inputs, rnn1_hc)
                rnn2_hc = self.rnn_layer2(rnn1_hc[0], rnn2_hc)
            else:
                rnn1_hc, rnn2_hc = self.rnn1_hc, self.rnn2_hc

            # decoder_state: batch_size * listener_hidden_dim
            decoder_state = rnn2_hc[0]
            # batch_size * value_dim
            context, attention = self.attention(decoder_state, listener_output, outputs_length)
            # batch_size * (speller_hiddem_dim + value_dim)
            concat_input = torch.cat((decoder_state, context), dim=1)
            # batch_size * class_size
            prob_linear = self.char_distribution_linear(concat_input)
            # batch_size * max_transcript_len
            prob_distribution = self.softmax(prob_linear)
            # 2d tensor
            index = torch.multinomial(prob_distribution, num_samples=1)
            preds = torch.squeeze(index)
            probs.append(prob_linear)
            predictions.append(preds)
            attentions.append(attention)

        # batch_size * timestep
        predictions = torch.stack(predictions, dim=1)
        # targets_for_loss: batch_size * timestep
        # targets length for loss: a list of len_for_loss (original_len - 1)
        return probs, predictions, targets_for_loss, targets_length_for_loss, attentions

    def inference(self, listener_output, outputs_length, timestep):
        predictions = []
        predictions.append(torch.tensor([LABEL_MAP['<sos>'] for i in range(len(listener_output))]))
        attentions = []

        for i in range(timestep):
            if i == 0:
                rnn1_hc, rnn2_hc = self.rnn1_hc, self.rnn2_hc
            else:
                embed_input = preds
                embed = self.embed(embed_input)
                inputs = torch.cat((embed, context), dim=1)
                rnn1_hc = self.rnn_layer1(inputs, rnn1_hc)
                rnn2_hc = self.rnn_layer2(rnn1_hc[0], rnn2_hc)

            # decoder_state: batch_size * listener_hidden_dim
            decoder_state = rnn2_hc[0]
            # batch_size * value_dim
            context, attention = self.attention(decoder_state, listener_output, outputs_length)
            # batch_size * (speller_hiddem_dim + value_dim)
            concat_input = torch.cat((decoder_state, context), dim=1)
            # batch_size * class_size
            prob_linear = self.char_distribution_linear(concat_input)
            # batch_size * max_transcript_len
            prob_distribution = self.softmax(prob_linear)
            # 2d tensor
            index = torch.multinomial(prob_distribution, num_samples=1)
            preds = torch.squeeze(index)
            predictions.append(preds)
            attentions.append(attention)

        # batch_size * timestep
        predictions = torch.stack(predictions, dim=1)
        predictions = predictions[:,1:] # delete <sos>
        prediction_list = []
        for i in range(len(predictions)):
            tmp = []
            for j in range(timestep):
                if predictions[i][j] == LABEL_MAP['<eos>']:
                    tmp = predictions[i][:j]
                    break
            if len(tmp) == 0:
                tmp = predictions[i]
            prediction_list.append(tmp)
        print(prediction_list)
        return prediction_list # a list of tensors


    def inference(self, listener_output, outputs_length, timestep):
        predictions = []
        predictions.append(torch.tensor([LABEL_MAP['<sos>'] for i in range(len(listener_output))]).to(DEVICE))
        attentions = []

        for i in range(timestep):
            if i == 0:
                rnn1_hc, rnn2_hc = self.rnn1_hc, self.rnn2_hc
            else:
                embed_input = preds
                embed = self.embed(embed_input)
                inputs = torch.cat((embed, context), dim=1)
                rnn1_hc = self.rnn_layer1(inputs, rnn1_hc)
                rnn2_hc = self.rnn_layer2(rnn1_hc[0], rnn2_hc)

            # decoder_state: batch_size * listener_hidden_dim
            decoder_state = rnn2_hc[0]
            # batch_size * value_dim
            context, attention = self.attention(decoder_state, listener_output, outputs_length)
            # batch_size * (speller_hiddem_dim + value_dim)
            concat_input = torch.cat((decoder_state, context), dim=1)
            # batch_size * class_size
            prob_linear = self.char_distribution_linear(concat_input)
            # batch_size * max_transcript_len
            prob_distribution = self.softmax(prob_linear)
            # 2d tensor
            index = torch.multinomial(prob_distribution, num_samples=1)
            preds = torch.squeeze(index)
            preds = preds.to(DEVICE)
            predictions.append(preds)
            attentions.append(attention)

        # batch_size * timestep
        predictions = torch.stack(predictions, dim=1)
        predictions = predictions[:,1:] # delete <sos>
        prediction_list = []
        for i in range(len(predictions)):
            tmp = []
            for j in range(timestep):
                if predictions[i][j] == LABEL_MAP['<eos>']:
                    tmp = predictions[i][:j]
                    break
            if len(tmp) == 0:
                tmp = predictions[i]
            prediction_list.append(tmp)
        print(prediction_list)
        return prediction_list # a list of tensors


class AttentionContext(nn.Module):
    def __init__(self, s_input_size, h_input_size, key_dim, value_dim):
        """
        input_size: listener_hidden_state * 4
        """
        super(AttentionContext, self).__init__()
        self.mlp_s = nn.Linear(in_features=s_input_size, out_features=key_dim)
        self.mlp_h = nn.Linear(in_features=h_input_size, out_features=key_dim)
        self.value_projection = nn.Linear(in_features=h_input_size, out_features=value_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, decoder_state, listener_output, outputs_length):
        """
        decoder_state: batch_size * decoder_hidden_dim
        listener_output: batch_size * longest_len * listener_output_dim
        """
        # query: batch_size * 1 * decoder_hidden_dim
        # after projection, query: batch_size * 1 * key_dim
        decoder_state = decoder_state.to(DEVICE)
        query = self.mlp_s(torch.unsqueeze(decoder_state, dim=1))
        # key: batch_size * key_dim * listener_output_dim
        listener_output = listener_output.to(DEVICE)
        key = self.mlp_h(listener_output)
        # energy: batch_size * 1 * listener_output_dim
        energy = torch.bmm(query, key.transpose(1,2))
        # attention: batch_size * 1 * listener_output_dim
        attention = self.softmax(energy)
        attention_mask = torch.ones(attention.shape).to(DEVICE)
        for i in range(listener_output.shape[0]):
            if i != 0:
                attention_mask[i][0][outputs_length[i]:] = 0
        attention = attention * attention_mask
        attention = F.normalize(attention, p=1, dim=2)
        # batch_size * listener_output_dim * value_dim
        value = self.value_projection(listener_output)
        utterance_mask = torch.ones(value.shape).to(DEVICE)
        for i in range(listener_output.shape[0]):
            if i != 0:
                utterance_mask[i][:][outputs_length[i]:] = 0
        value = value * utterance_mask
        # context: batch_size * 1 * value_dim
        context = torch.bmm(attention, value)
        # context: batch_size * value_dim
        context = torch.squeeze(context, dim=1)
        return context, attention

class LAS(nn.Module):
    def __init__(self, input_size, listener_hidden_size, nlayers,
                 speller_hidden_dim, embedding_dim,
                 class_size, key_dim, value_dim, batch_size):
        super(LAS, self).__init__()
        self.listener = Listener(input_size=40, hidden_size=listener_hidden_size, nlayers=4)
        self.listener = self.listener.to(DEVICE)
        self.speller = Speller(listener_hidden_size*2, speller_hidden_dim,
                               embedding_dim, class_size, key_dim, value_dim,
                               batch_size)
        self.speller = self.speller.to(DEVICE)

    def forward(self, inputs, targets, teacher_forcing):
        listener_outputs, outputs_length = self.listener(inputs)
        probs, predictions, targets_for_loss, targets_length_for_loss, attentions = self.speller(listener_outputs, outputs_length, targets, teacher_forcing)

        return probs, predictions, targets_for_loss, targets_length_for_loss, attentions

    def inference(self, inputs, targets):
        listener_outputs, outputs_length = self.listener(inputs)
        prediction_list = self.speller.inference(listener_outputs, outputs_length, timestep=300)
        return prediction_list


if __name__ == "__main__":
    u1 = torch.randn((30,40))
    u2 = torch.randn((20,40))
    # u3 = torch.randn((10,40))
    t1 = torch.randint(0,32,(9,))
    t2 = torch.randint(0,32,(8,))
    # t3 = torch.randint(0,32,(7,))
    inputs = [u1, u2]
    targets = [t1, t2]
    listener_model = Listener(40, 256, 4)
    padded_outputs, outputs_length = listener_model(inputs)

    # decoder_state = torch.rand((3,200))
    # attention_model = AttentionContext(200, 512, 256, 500)
    # context = attention_model(decoder_state, padded_outputs, outputs_length)

    speller_model = Speller(listener_hidden_dim=512, speller_hidden_dim=512,
                            embedding_dim=256, class_size=34, key_dim=128, value_dim=128,
                            batch_size=2)
    # speller_model(padded_outputs, outputs_length, targets, 0.9)
    speller_model.inference(padded_outputs, outputs_length, 10)

    # las = LAS(input_size=40, listener_hidden_size=256, nlayers=4,
    #           speller_hidden_dim=512, embedding_dim=256,
    #           class_size=34, key_dim=128, value_dim=128, batch_size=2)
    # las(inputs, targets, 0.9)



