import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os

from myDataset import myDataset, collate_seq
from config import MODEL_CONFIG as CONF
from model import LAS
from vocab import NUM_2_CHAR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

def train(train_loader, model, optimizer, criterion, epoch):
    loss_sum = 0
    perplexity_sum = 0
    for step, (inputs, targets) in enumerate(train_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        probs, predictions, targets_for_loss, targets_length_for_loss, \
        attentions = model(inputs, targets, teacher_forcing=0.9)

        loss = 0
        for i in range(len(probs)):
            loss = loss + criterion(probs[i], targets_for_loss[:, i])

        loss.backward()
        optimizer.step()
        perplexity_sum = perplexity_sum + np.exp(loss.item() / len(inputs) / max(targets_length_for_loss))
        if step % 10 ==0:
            print("epoch {}, step {}, loss per step {}, perplexity {}, finish {}".format(
                epoch, step, loss/len(inputs), perplexity_sum, (step+1)*len(inputs)))
        perplexity_sum = 0
        if (step+1) % args.checkpoint == 0:
            save_model(epoch, model, optimizer, loss, step, "./weights/")

def dev(dev_loader, model, optimizer, criterion):
    for step, (inputs, targets) in enumerate(dev_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        prediction_list = model.inference(inputs, targets)
        print(prediction_list)
        batch_size = len(prediction_list)
        for i in range(batch_size):
            pred = ""
            for j in range(len(prediction_list[i])):
                pred += NUM_2_CHAR[int(prediction_list[i][j].to("cpu"))]
            print(pred)



def save_model(epoch, model, optimizer, loss, step, save_path):
    filename = save_path + str(epoch) + '-' + str(step) + '-' + "%.6f" % loss.item() + '.pth'
    print('Save model at Train Epoch: {} [Step: {}\tLoss: {:.12f}]'.format(
        epoch, step, loss.item()))
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filename)

def load_model(epoch, step, loss, model, optimizer, save_path):
    filename = save_path + str(epoch) + '-' + str(step) + '-' + str(loss) + '.pth'
    if os.path.isfile(filename):
        print("######### loading weights ##########")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        print('########## loading weights done ##########')
        return model, optimizer, start_epoch, loss
    else:
        print("no such file: ", filename)

def main(args):
    # Load configuration
    input_size = CONF["input_size"]
    listener_hidden_size = CONF["listener_hidden_size"]
    nlayers = CONF["nlayers"]
    speller_hidden_dim = CONF["speller_hidden_dim"]
    embedding_dim = CONF["embedding_dim"]
    class_size = CONF["class_size"]
    key_dim = CONF["key_dim"]
    value_dim = CONF["value_dim"]
    batch_size = CONF["batch_size"]

    train_path = "./data/dev.npy"
    train_transcripts_path = "./data/dev_char.npy"
    train_set = myDataset(train_path, train_transcripts_path)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, collate_fn=collate_seq, num_workers=4)

    dev_path = "./data/dev.npy"
    dev_transcripts_path = "./data/dev_char.npy"
    dev_set = myDataset(dev_path, dev_transcripts_path)
    dev_loader = DataLoader(dev_set, shuffle=False, batch_size=batch_size, collate_fn=collate_seq, num_workers=4)

    # test_path = "./data/test.npy"
    # dev_set = myDataset(dev_path, None)
    # dev_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, collate_fn=collate_seq, num_workers=4)

    model = LAS(input_size, listener_hidden_size, nlayers,
                speller_hidden_dim, embedding_dim,
                class_size, key_dim, value_dim, batch_size)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    start_epoch = 0
    nepochs = args.epochs
    if args.resume is True:
            model, optimizer, start_epoch, loss = load_model(
                args.load_epoch,
                args.load_step,
                args.load_loss,
                model,
                optimizer,
                "./weights/"
            )
    for epoch in range(start_epoch, nepochs):
        model.train()
        train(train_loader, model, optimizer, criterion, epoch)
        # model.eval()
        # eval()
    model.eval()
    dev(dev_loader, model, optimizer, criterion)

def arguments():
    parser = argparse.ArgumentParser(description="LAS")
    # parameters for training process
    parser.add_argument('--epochs', type=int, default=20, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='L2 regularization')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help="learning rate")
    parser.add_argument('--checkpoint', type=int, default=900, metavar="R",
                        help='checkpoint to save model parameters')
    parser.add_argument('--resume', type=bool, default=False, metavar="R",
                        help='resume training from saved weight')
    parser.add_argument('--weights-path', type=str, default="./weights/",
                        help='path to save weights')
    parser.add_argument('-load-epoch', type=str, default=0, metavar="LE",
                        help='number of epoch to be loaded')
    parser.add_argument('-load-step', type=str, default=0, metavar="LS",
                        help='number of step to be loaded')
    parser.add_argument('-load-loss', type=str, default=0, metavar="LL",
                        help='loss item to be loaded')

    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    main(args)
