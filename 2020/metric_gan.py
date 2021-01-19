import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms as trans
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
        self.bi_lstm = nn.LSTM(num_feats, hidden_size=200, num_layers=2, 
                               batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(200*2, 300)
        self.fc2 = nn.Linear(300, 257)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.real
        x, (h, _) = self.bi_lstm(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.sigmoid(self.fc2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=15,
                                 kernel_size=(5, 5))
        self.conv2d2 = nn.Conv2d(in_channels=15, out_channels=25,
                                 kernel_size=(7, 7))
        self.conv2d3 = nn.Conv2d(in_channels=25, out_channels=40,
                                 kernel_size=(9, 9))
        self.conv2d4 = nn.Conv2d(in_channels=40, out_channels=50,
                                 kernel_size=(11, 11))
        self.avg_pool = nn.AvgPool2d(kernel_size=(50,50))
        self.leaky_relu = nn.LeakyReLU()
        #self.fc1 = nn.Linear()
    def forward(self, x):
        x = x.real
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = self.conv2d4(x)
        return x


device = torch.device("cuda:0")
generator = Generator(513)
generator.cuda()
generator = generator.to(device)

discriminator = Discriminator()
discriminator = discriminator.to(device)

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
    y = predict(x.squeeze(1), (out_r, out_i), floor=True)
    y = torch.transpose(y, 1, 2)
    y_gen = generator(y)
    y_gen = torch.transpose(y_gen, 1, 2).unsqueeze(1)
    print(y_gen.shape)
    y_disc = discriminator(y_gen)
    print(y_disc.shape)