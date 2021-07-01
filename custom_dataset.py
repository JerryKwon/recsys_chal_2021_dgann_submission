"""
Name: custom_dataset.py
Description: Custom Dataset for Neural Network model
"""

import torch
from torch.utils.data import Dataset

class DANDataset(Dataset):
    def __init__(self, np_array, np_idx_len):
        super(DANDataset, self).__init__()
        self.np_array = np_array
        self.np_idx_len = np_idx_len
        # self.label = label

    def __getitem__(self, index):
        return torch.LongTensor(self.np_array[index]), self.np_idx_len[index]

    def __len__(self):
        return self.np_array.shape[0]

class AnnDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.x_data = X
        # self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        # y = self.y_data[idx]

        return x