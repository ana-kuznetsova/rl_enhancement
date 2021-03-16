import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import numpy as np
import os
import copy
import soundfile as sf
from pystoi import stoi
from pypesq import pesq
from tqdm import tqdm
import librosa

from preproc import Data, DataTest
from preproc import collate_custom
from modules import Actor, init_weights, predict
from losses import ES_MSE


def train(clean_path, noisy_path, model_path, num_epochs):
    device = torch.device("cuda:2")
    model = Actor()
    model.cuda()
    model = model.to(device)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=[2, 3])


    criterion = ES_MSE()
    criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    losses = []
    val_losses = []
    best = copy.deepcopy(model.state_dict())
    prev_val=99999

    for epoch in range(1, num_epochs+1):
        epoch_loss = 0

        dataset = Data(clean_path, noisy_path, 1000)
        loader = data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_custom)

        model.train()

        for batch in loader:
            x = batch["noisy"].unsqueeze(1).to(device)
            t = batch["clean"].unsqueeze(1).to(device)
            m = batch["mask"].to(device)
            out_r, out_i = model(x)
            print(out_r.shape)
            y = predict(x.squeeze(1), (out_r, out_i))
            print("predict shape:", y.shape)
            t = t.squeeze()
            m = m.squeeze()
            x = x.squeeze()


train('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
     '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
     '/nobackup/anakuzne/data/experiments/speech_enhancement/es/es_0/', 100)