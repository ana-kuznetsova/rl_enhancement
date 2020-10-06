import torch.utils.data as data
import numpy as np
import torch


class QDataSet(data.Dataset):
    def __init__(self, batch_indices):
        self.x = np.random.rand(10, 2)
        self.y = np.random.rand(10, 2)
        self.batch_indices = batch_indices
    def __getitem__(self, index):
        start_idx = self.batch_indices[index]
        end_idx = self.batch_indices[index+1]
        return torch.from_numpy(self.x[start_idx:end_idx]).float(), torch.from_numpy(self.y[start_idx:end_idx]).float()
    def __len__(self):
        return len(self.batch_indices) - 1

dataset = QDataSet([0, 2, 5, 9])
loader = data.DataLoader(dataset, batch_size=1)

for i, data in enumerate(loader):
    x = data[0]
    y = data[1]
    print(x.shape, y.shape)