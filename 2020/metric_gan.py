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
from model import Actor
from model import predict, inverse


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


device = torch.device("cuda:0")
critic = Generator(513)
critic.cuda()
critic = critic.to(device)

actor = Actor()
actor.load_state_dict(torch.load('/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor/actor_best.pth'))
actor = actor.to(device)

dataset = DataLoader('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
                     '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', get_feats, 1000)
loader = data.DataLoader(dataset, batch_size=10, shuffle=True)

for i, batch in enumerate(loader):
    x = batch["noisy"].unsqueeze(1).to(device)
    t = batch["clean"].unsqueeze(1).to(device)
    m = batch["mask"].to(device)
    out_r, out_i = actor(x)
    out_r = torch.transpose(out_r, 1, 2)
    out_i = torch.transpose(out_i, 1, 2)
    y = predict(x.squeeze(1), (out_r, out_i))
    y = critic(y.unsqueeze(1))
    print(y)