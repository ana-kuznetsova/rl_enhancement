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


def train(clean_path, noisy_path, model_path, num_epochs, elite_size=200):
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
        dataset = Data(clean_path, noisy_path, 50)
        loader = data.DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_custom)

        model.train()
        
        individual_losses = []

        for i, batch in enumerate(loader):
            print('Step:{}/{}'.format(i, len(loader)))
            x = batch["noisy"].unsqueeze(1).to(device)
            t = batch["clean"].unsqueeze(1).to(device)
            out_r, out_i = model(x)
            out_r = torch.transpose(out_r, 1, 2)
            out_i = torch.transpose(out_i, 1, 2)
            y = predict(x.squeeze(1), (out_r, out_i))
            t = t.squeeze()
            batch_losses = criterion(torch.abs(y), torch.abs(t))
            individual_losses.extend(batch_losses)
        
        ### Select elite set and backpropagate from N best ###
        elite_set = [(i, (individual_losses[i])) for i in range(50)]
        elite_set = sorted(elite_set, key=lambda x:x[1])[:elite_size]
        print("elite set:", elite_set, torch.tensor([i[1] for i in elite_set]))
        elite_set_loss = torch.mean(torch.tensor([i[1] for i in elite_set]))
        
        optimizer.zero_grad()
        elite_set_loss.backward()
        optimizer.step()

        losses.append(elite_set_loss)
        np.save(os.path.join(model_path, "elite_loss_train.npy"), np.array(losses))
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, elite_set_loss.detach().numpy().cpu()))
            


train('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
     '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
     '/nobackup/anakuzne/data/experiments/speech_enhancement/es/es_0/', 100, 10)