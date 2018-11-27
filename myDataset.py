import torch
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class myDataset(Dataset):
    def __init__(self, data_path, transcripts_path):
        self.data = np.load(data_path, encoding='bytes')[:1100]
        self.flag = False
        if transcripts_path != None:
            self.label = np.load(transcripts_path)
            self.flag = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        sequence = torch.tensor(sequence)
        if self.flag is True:
            target = self.label[index]
            target = torch.tensor(target)
            return sequence, target
        else:
            return sequence, [-1]

def collate_seq(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs, targets


if __name__ == '__main__':
    data_path = "./data/dev.npy"
    transcripts_path = "./data/dev_char.npy"
    train_set = myDataset(data_path, transcripts_path)
    # for i in range(3):
    #     sequence, targets = train_set.__getitem__(i)
    #     print(sequence.shape) # seq_len * 40
    train_loader = DataLoader(train_set, shuffle=False, batch_size=4, collate_fn=collate_seq, num_workers=4)
    for step, (inputs, targets) in enumerate(train_loader):
        if step == 0:
            print(inputs[0].shape, inputs[1].shape)
            print(targets)



