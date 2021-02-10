import sys
sys.path.insert(1, '../2020/')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os
import copy
import soundfile as sf

from pypesq import pesq
from tqdm import tqdm
import librosa

from modules import Actor, init_weights, predict, inverse
from losses import SDRLoss
from preproc import collate_custom


def generate_curriculum(clean_path, noisy_path, model_path):
    fnames = os.listdir(clean_path)
    d = {"fname":[], "pesq":[]}
    for fname in tqdm(fnames):
        wav_noisy, sr = librosa.core.load(os.path.join(noisy_path, fname), sr=16000)
        wav_clean, sr = librosa.core.load(os.path.join(clean_path, fname), sr=16000)
        score = pesq(wav_clean, wav_noisy)
        d["fname"].append(fname)
        d["pesq"].append(score)
    df = pd.DataFrame.from_dict(d)
    df = df.sort_values(by=['pesq'])
    df.to_csv(os.path.join(model_path, "train_sort.tsv"), sep='\t')


class Data(data.Dataset):
    def __init__(self, clean_path, noisy_path, csv_path):
        self.fnames = pd.read_csv(csv_path)['fname']
        self.clean_paths = [os.path.join(clean_path, f) for f in self.fnames]
        self.noisy_paths = [os.path.join(noisy_path, f) for f in self.fnames]

    def __len__(self):
        return len(self.clean_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.clean_paths[idx], self.noisy_paths[idx])
        return sample


def train_curriculum(clean_path, noisy_path, model_path, num_epochs):
    device = torch.device("cuda:1")
    model = Actor()
    model.cuda()
    model = model.to(device)
    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=[1, 2])

    criterion = SDRLoss()
    criterion.cuda()

    losses = []
    val_losses = []
    best = copy.deepcopy(model.state_dict())
    prev_val=99999


    dataset = Data(clean_path, noisy_path, os.path.join(model_path, 'train.tsv'))
    loader = data.DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_custom)

    for epoch in range(1, num_epochs+1):
        if epoch <= 100:
            lr = 0.0001
        else:
            lr = lr/100

        optimizer = optim.Adam(model.parameters(), lr=lr)

        epoch_loss = 0

        for batch in loader:
            x = batch["noisy"].unsqueeze(1).to(device)
            t = batch["clean"].unsqueeze(1).to(device)
            m = batch["mask"].to(device)
            out_r, out_i = model(x)
            out_r = torch.transpose(out_r, 1, 2)
            out_i = torch.transpose(out_i, 1, 2)
            y = predict(x.squeeze(1), (out_r, out_i))
            t = t.squeeze()
            m = m.squeeze()
            targets, preds = inverse(t, y, m)
            loss = criterion(targets, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            epoch_loss+=loss

            losses.append(epoch_loss/len(loader))
        np.save(os.path.join(model_path, "loss_actor_curr.npy"), np.array(losses))
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))


        if epoch%5==0:
            ##Validation
            overall_val_loss = 0

            dataset = Data(clean_path, noisy_path, os.path.join(model_path, 'dev.tsv'))
            loader = data.DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_custom)

            for batch in loader:
                x = batch["noisy"].unsqueeze(1).to(device)
                t = batch["clean"].unsqueeze(1).to(device)
                m = batch["mask"].to(device)
                out_r, out_i = model(x)
                out_r = torch.transpose(out_r, 1, 2)
                out_i = torch.transpose(out_i, 1, 2)
                y = predict(x.squeeze(1), (out_r, out_i))
                t = t.squeeze()
                m = m.squeeze()
                targets, preds = inverse(t, y, m)
                loss = criterion(targets, preds)
                overall_val_loss+=loss.detach().cpu().numpy()

                curr_val_loss = overall_val_loss/len(loader)
            val_losses.append(curr_val_loss)
            print('Validation loss: ', curr_val_loss)
            np.save(os.path.join(model_path, 'val_loss_actor_curr.npy'), np.array(val_losses))

            if curr_val_loss < prev_val:
                torch.save(best, os.path.join(model_path, 'actor_best.pth'))
                prev_val = curr_val_loss

            torch.save(best, os.path.join(model_path, "actor_last.pth"))


train_curriculum('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
                '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
                '/nobackup/anakuzne/data/experiments/speech_enhancement/curriculum/', 150)