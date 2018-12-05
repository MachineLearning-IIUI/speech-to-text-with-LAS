import pdb
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from vocab import LABEL_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Listener(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers):
        super(Listener, self).__init__()
        self.input_size = input_size
        self.nlayers = nlayers
        lstm_list = []
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

            lstm_list.append(lstm)
        self.lstm_list = nn.ModuleList(lstm_list)

    def forward(self, inputs_list): # batch_size * var_seq_len * 40
        batch_size = len(inputs_list)

        inputs_length = [len(utterance) for utterance in inputs_list] # original utterance lengths
        inputs_length = torch.LongTensor(inputs_length)
        outputs_length = inputs_length // 8 # output utterance lengths
        
        # packed_inputs.data.shape: (sum_len * 40)
        packed_inputs = rnn.pack_sequence(inputs_list)

        for i in range(self.nlayers):
            # lstm_output: sum_len * (hidden_size*2)
            if i == 0:
                lstm_outputs, _ = self.lstm_list[i](packed_inputs)
            else:
                lstm_outputs, _ = self.lstm_list[i](packed_inputs)
            
            # unpacked_outputs shape: max_len * batch_size * (hidden_size*2)
            unpacked_outputs, _ = rnn.pad_packed_sequence(lstm_outputs)

            if i != self.nlayers - 1:
                longest_len = unpacked_outputs.shape[0]
                dim = unpacked_outputs.shape[2]
                # transpose lstm output to batch_size * longest_len * dim
                unpacked_outputs = unpacked_outputs.permute(1, 0, 2)
                # chop off the extra
                if longest_len % 2 != 0:
                    unpacked_outputs = unpacked_outputs[:,0:-1,...]
                longest_len = longest_len // 2
                dim = dim * 2
                # reshape to batch_size * (longest_len/2) * (dim*2)
                unpacked_outputs = unpacked_outputs.contiguous().view(-1, longest_len, dim)
                # transpose back to (longest_len/2) * batch_size * (dim*2)
                unpacked_outputs = unpacked_outputs.permute(1, 0, 2)
                lengths = inputs_length // (2 ** (i+1))
                packed_inputs = rnn.pack_padded_sequence(unpacked_outputs, lengths)

        # batch_size * longest_len * (hidden_size*2)
        unpacked_outputs = unpacked_outputs.transpose(0, 1)
        # outputs_length is a 1d tensor
        return unpacked_outputs, outputs_length

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

        self.rnn1_hidden_state = nn.Parameter(torch.zeros(batch_size, speller_hidden_dim)).to(DEVICE)
        self.rnn1_cell_state = nn.Parameter(torch.zeros(batch_size, speller_hidden_dim)).to(DEVICE)
        self.rnn2_hidden_state = nn.Parameter(torch.zeros(batch_size, speller_hidden_dim)).to(DEVICE)
        self.rnn2_cell_state = nn.Parameter(torch.zeros(batch_size, speller_hidden_dim)).to(DEVICE)
        self.speller_hidden_dim = speller_hidden_dim

    def forward(self, listener_output, outputs_length, targets, teacher_forcing):
        if self.rnn1_hidden_state.shape[0] != len(listener_output):
            batch_size = len(listener_output)
            self.rnn1_hidden_state = nn.Parameter(torch.zeros(batch_size, self.speller_hidden_dim)).to(DEVICE)
            self.rnn1_cell_state = nn.Parameter(torch.zeros(batch_size, self.speller_hidden_dim)).to(DEVICE)
            self.rnn2_hidden_state = nn.Parameter(torch.zeros(batch_size, self.speller_hidden_dim)).to(DEVICE)
            self.rnn2_cell_state = nn.Parameter(torch.zeros(batch_size, self.speller_hidden_dim)).to(DEVICE)

        targets_length_for_loss = [len(transcript)-1 for transcript in targets] # original transcript length - 1
        timestep = max(targets_length_for_loss) # max_transcript_len - 1
        # batch_size * max_transcript_len (LongTensor)
        padded_targets = rnn.pad_sequence(targets, batch_first=True).long().to(DEVICE)
        targets_for_loss = rnn.pad_sequence(targets, batch_first=True, padding_value=-1).long().to(DEVICE)
        targets_for_loss = targets_for_loss[:,1:] # only need targets starting from index 1

        probs = []
        predictions = []
        attentions = []

        rnn1_h = self.rnn1_hidden_state
        rnn1_c = self.rnn1_cell_state
        rnn2_h = self.rnn2_hidden_state
        rnn2_c = self.rnn2_cell_state
        context, attention = self.attention(rnn2_h, listener_output, outputs_length)

        for i in range(timestep):
            if i != 0:
                if np.random.random() < teacher_forcing:
                    # embed input is a 1d tensor
                    # batch_size * embedding_dim
                    embed = self.embed(padded_targets[:,i])
                else:
                    embed = self.embed(preds)
                inputs = torch.cat((embed, context), dim=1)
                rnn1_h, rnn1_c = self.rnn_layer1(inputs, (rnn1_h, rnn1_c))
                rnn2_h, rnn2_c = self.rnn_layer2(rnn1_h, (rnn2_h, rnn2_c))
            else: # i == 0
                embed = self.embed(padded_targets[:,0])
                inputs = torch.cat((embed, context), dim=1)
                rnn1_h, rnn1_c = self.rnn_layer1(inputs, (rnn1_h, rnn1_c))
                rnn2_h, rnn2_c = self.rnn_layer2(rnn1_h, (rnn2_h, rnn2_c))

            # decoder_state: batch_size * listener_hidden_dim
            decoder_state = rnn2_h
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
        if self.rnn1_hidden_state.shape[0] != len(listener_output):
            batch_size = len(listener_output)
            self.rnn1_hidden_state = nn.Parameter(torch.zeros(batch_size, self.speller_hidden_dim)).to(DEVICE)
            self.rnn1_cell_state = nn.Parameter(torch.zeros(batch_size, self.speller_hidden_dim)).to(DEVICE)
            self.rnn2_hidden_state = nn.Parameter(torch.zeros(batch_size, self.speller_hidden_dim)).to(DEVICE)
            self.rnn2_cell_state = nn.Parameter(torch.zeros(batch_size, self.speller_hidden_dim)).to(DEVICE)

        predictions = []
        preds = torch.tensor([LABEL_MAP['<sos>'] for i in range(len(listener_output))]).to(DEVICE)
        attentions = []

        rnn1_h = self.rnn1_hidden_state
        rnn1_c = self.rnn1_cell_state
        rnn2_h = self.rnn2_hidden_state
        rnn2_c = self.rnn2_cell_state
        context, attention = self.attention(rnn2_h, listener_output, outputs_length)

        for i in range(timestep):
            if i == 0:
                embed_input = preds
                embed = self.embed(preds)
                inputs = torch.cat((embed, context), dim=1)
                rnn1_h, rnn1_c = self.rnn_layer1(inputs, (rnn1_h, rnn1_c))
                rnn2_h, rnn2_c = self.rnn_layer2(rnn1_h, (rnn2_h, rnn2_c))
            else:
                embed_input = preds
                embed = self.embed(embed_input)
                inputs = torch.cat((embed, context), dim=1)
                rnn1_h, rnn1_c = self.rnn_layer1(inputs, (rnn1_h, rnn1_c))
                rnn2_h, rnn2_c = self.rnn_layer2(rnn1_h, (rnn2_h, rnn2_c))

            # decoder_state: batch_size * listener_hidden_dim
            decoder_state = rnn2_h
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
        # predictions = predictions[:,1:] # delete <sos>
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
        decoder_state = decoder_state.to(DEVICE)
        # query: batch_size * 1 * decoder_hidden_dim
        query = torch.unsqueeze(decoder_state, dim=1)
        # after projection, query: batch_size * 1 * key_dim
        query = self.mlp_s(query)

        listener_output = listener_output.to(DEVICE)
        # key: batch_size * key_dim * listener_output_dim
        key = self.mlp_h(listener_output)
        key = key.transpose(1, 2)
        # energy: batch_size * 1 * listener_output_dim
        energy = torch.bmm(query, key)
        # attention: batch_size * 1 * listener_output_dim
        attention = self.softmax(energy)
        attention_mask = torch.ones(attention.shape).to(DEVICE)
        for i in range(listener_output.shape[0]):
            if i != 0:
                attention_mask[i, :, outputs_length[i]:] = 0
        attention_mask = Variable(attention_mask)
        attention = attention * attention_mask
        attention = F.normalize(attention, p=1, dim=2)
        # batch_size * listener_output_dim * value_dim
        value = self.value_projection(listener_output)
        # utterance_mask = torch.ones(value.shape).to(DEVICE)
        # for i in range(listener_output.shape[0]):
        #     if i != 0:
        #         utterance_mask[i][:][outputs_length[i]:] = 0
        # value = value * utterance_mask

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
    u3 = torch.randn((15,40))
    t1 = torch.randint(0,32,(9,))
    t2 = torch.randint(0,32,(8,))
    t3 = torch.randint(0,32,(7,))
    inputs = [u1, u2, u3]
    targets = [t1, t2, t3]
    listener_model = Listener(40, 256, 4)
    padded_outputs, outputs_length = listener_model(inputs)

    # decoder_state = torch.rand((3,200))
    # attention_model = AttentionContext(200, 512, 256, 500)
    # context = attention_model(decoder_state, padded_outputs, outputs_length)

    # speller_model = Speller(listener_hidden_dim=512, speller_hidden_dim=512,
    #                         embedding_dim=256, class_size=34, key_dim=128, value_dim=128,
    #                         batch_size=3)
    # speller_model(padded_outputs, outputs_length, targets, 0.9)
    # speller_model.inference(padded_outputs, outputs_length, 10)

    las = LAS(input_size=40, listener_hidden_size=256, nlayers=4,
              speller_hidden_dim=512, embedding_dim=256,
              class_size=34, key_dim=128, value_dim=128, batch_size=3)
    las(inputs, targets, 0.9)
    # las.inference(inputs, targets)
