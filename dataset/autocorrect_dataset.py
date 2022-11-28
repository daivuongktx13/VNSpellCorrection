import torch
import numpy as np

class SpellCorrectDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
    def take(self, n = 1):
        indies = np.random.choice(len(self.dataset), n)
        return [self.dataset[idx] for idx in indies]
