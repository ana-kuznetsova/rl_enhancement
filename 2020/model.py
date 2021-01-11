import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from preproc import DataLoader
from preproc import get_feats

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(in_channels=3, out_channels=30,
                                 kernel_size=(5, 15), stride=(1,1), 
                                 padding=(2,7))
        self.conv2d2 = nn.Conv2d(in_channels=30, out_channels=60,
                                 kernel_size=(5, 15), stride=(1,1), 
                                 padding=(2,7))
        #self.conv1d3 = nn.Conv1d(in_channels=60, out_channels=1)

    def forward(self, x):
        pass

device = torch.device("cuda")
model = Actor()
model.cuda()
model = model.to(device)

dataset = DataLoader()

loader = data.DataLoader(dataset, batch_size=10, shuffle=True)