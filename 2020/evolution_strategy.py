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


def train(clean_path, noisy_path, model_path, num_epochs, elite_size=200, population=1000, val_size=300):
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
        dataset = Data(clean_path, noisy_path, population)
        loader = data.DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_custom)

        model.train()
        
        individual_losses = []
        print("Steps:", len(loader))

        for i, batch in enumerate(loader):
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
        elite_set = [(i, (individual_losses[i])) for i in range(population)]
        elite_set = sorted(elite_set, key=lambda x:x[1])[:elite_size]
        elite_set_loss = torch.sum(torch.tensor([i[1] for i in elite_set]))/elite_size
        elite_set_loss.requires_grad=True
        
        optimizer.zero_grad()
        elite_set_loss.backward()
        optimizer.step()

        losses.append(elite_set_loss)
        np.save(os.path.join(model_path, "elite_loss_train.npy"), np.array(losses))
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, elite_set_loss.detach().cpu().numpy()))


        if epoch%5==0:
            ##Validation
            model.eval()
            val_loss = 0

            print("Validation...")
            dataset = Data(clean_path, noisy_path, val_size)
            loader = data.DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_custom)
            print("Steps:", len(loader))

            for i, batch in enumerate(loader):
                x = batch["noisy"].unsqueeze(1).to(device)
                t = batch["clean"].unsqueeze(1).to(device)
                out_r, out_i = model(x)
                out_r = torch.transpose(out_r, 1, 2)
                out_i = torch.transpose(out_i, 1, 2)
                y = predict(x.squeeze(1), (out_r, out_i))
                t = t.squeeze()
                batch_losses = criterion(torch.abs(y), torch.abs(t))
                batch_losses = torch.sum(batch_losses/len(batch_losses)).detach().cpu().numpy()
                val_loss+=batch_losses
            val_losses.append(val_loss/len(loader))
            print('Validation loss:', val_loss)
            np.save(os.path.join(model_path, 'val_loss.npy'), np.array(val_losses))


            if  val_loss < prev_val:
                torch.save(best, os.path.join(model_path, 'es_best.pth'))
                prev_val = val_loss
            torch.save(best, os.path.join(model_path, "es_last.pth"))
        
            
            


train('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
     '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
     '/nobackup/anakuzne/data/experiments/speech_enhancement/es/es_0/', 100)