import torch
from torch.utils.data import Dataset
import numpy as np

class myDataset(Dataset):
    def __init__(self, data_path, transcripts_path):
        self.data = np.load(data_path, encoding='bytes')
        if transcripts_path != None:
            self.label = np.load(transcripts_path)
        else:
            self.label = None
    
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, index):
        sequence = self.data[index]
        sequence = torch.tensor(sequence)
        if self.label != None:
            target = self.label[index]
            target = torch.tensor(target)
            return sequence, target
        else:
            return sequence


        