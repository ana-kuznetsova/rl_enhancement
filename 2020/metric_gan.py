import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
import copy
import soundfile as sf

from preproc import DataLoader
from preproc import get_feats


class Generator(nn.Module):
    def __init__(self, num_feats):
        super().__init__()
        self.bnorm = nn.BatchNorm2d(num_feats)
        self.bi_lstm = nn.LSTM(num_feats, hidden_size=200, num_layers=2, 
                               batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(200*2, 300)
        self.fc2 = nn.Linear(300, 257)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bnorm(x)
        x = self.bi_lstm(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

dataset = DataLoader('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
                     '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', get_feats, 1000)
loader = data.DataLoader(dataset, batch_size=5, shuffle=True)

for batch in loader:
    x = batch["noisy"].unsqueeze(1)
    print(x.shape)