import torch.utils.data as data
import numpy as np
import torch


class QDataSet(data.Dataset):
    def __init__(self):
        self.x = np.random.rand(10, 2)
        self.y = np.random.rand(10, 2)
        self.batch_indices = batch_indices
    def __getitem__(self, index):
        start_idx = self.batch_indices[index]
        end_idx = self.batch_indices[index+1]
        return torch.from_numpy(self.x[start_idx:end_idx]).float(), torch.from_numpy(self.y[start_idx:end_idx]).float()
    def __len__(self):
        #Number of files
        return len(self.batch_indices) - 1

dataset = QDataSet()
loader = data.DataLoader(dataset, batch_size=1)

for data in loader:
    data = data.view(-1, 1)
    print(data.shape)