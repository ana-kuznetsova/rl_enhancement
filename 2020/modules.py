import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import numpy as np
import os
import copy
import soundfile as sf
from pystoi import stoi
from pypesq import pesq
from tqdm import tqdm
import librosa

from preproc import Data, DataTest
from preproc import get_feats, collate_custom
from losses import SDRLoss, CriticLoss

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=30,
                                 kernel_size=(5, 15), stride=(1,1), 
                                 padding=(2,7))
        self.conv2d2 = nn.Conv2d(in_channels=30, out_channels=60,
                                 kernel_size=(5, 15), stride=(1,1), 
                                 padding=(2,7))
        self.conv2d3 = nn.Conv2d(in_channels=60, out_channels=1,
                                 kernel_size=(1, 1), stride=(1,1))
        self.linear1 = nn.Linear(257, 512)
        self.bi_lstm = nn.LSTM(512, hidden_size=512, num_layers=2, 
                               batch_first=True, dropout=0.3, bidirectional=True)
        self.linear2 = nn.Linear(1024, 257*2)

    def forward(self, x):
        x = x.real
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = torch.transpose(x.squeeze(), 1, 2)
        x = self.linear1(x)
        x, (h, _) = self.bi_lstm(x)
        x = self.linear2(x)

        real = []
        imag = []
        
        for m in x:
            r = m[:,:int(m.shape[1]/2)]
            real.append(r)
            i = m[:,int(m.shape[1]/2):]
            imag.append(i)

        return torch.stack(real), torch.stack(imag)
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.bnorm = nn.BatchNorm1d(257)
        self.conv2d1 = spectral_norm(nn.Conv2d(in_channels=1, out_channels=15,
                                 kernel_size=(5, 5)))
        self.conv2d2 = spectral_norm(nn.Conv2d(in_channels=15, out_channels=25,
                                 kernel_size=(7, 7)))
        self.conv2d3 = spectral_norm(nn.Conv2d(in_channels=25, out_channels=40,
                                 kernel_size=(9, 9)))
        self.conv2d4 = spectral_norm(nn.Conv2d(in_channels=40, out_channels=50,
                                 kernel_size=(11, 11)))
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = spectral_norm(nn.Linear(50, 50))
        self.fc2 = spectral_norm(nn.Linear(50, 10))
        self.out = spectral_norm(nn.Linear(10, 1))

    def forward(self, x):
        x = x.real
        x = self.bnorm(x).unsqueeze(1)
        x = self.leaky_relu(self.conv2d1(x))
        x = self.leaky_relu(self.conv2d2(x))
        x = self.leaky_relu(self.conv2d3(x))
        x = self.leaky_relu(self.conv2d4(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x

def predict(x, model_out, floor=False):
    def floor_mask(model_out, treshold=0.05):
        temp = model_out[0]
        temp[temp < treshold] = treshold
        return [temp, model_out[1]]
    if floor:
        model_out = floor_mask(model_out)
    temp = torch.complex(model_out[0], model_out[1])
    return x*temp

def inverse(t, y , m, device, x=None):    
    def normalize(v):
        #return v/np.linalg.norm(v)
        return v/torch.norm(v)

    targets = []
    preds = []
    source = []

    for i in range(t.shape[0]):
        pad_idx = int(torch.sum(m[i]))
        if x==None:
            t_i = t[i]
            #t_i = t_i[:, :pad_idx].detach().cpu().numpy()
            t_i = t_i[:, :pad_idx]
            y_i = y[i]
            #y_i = y_i[:, :pad_idx].detach().cpu().numpy()
            y_i = y_i[:, :pad_idx]
            #t_i = librosa.core.istft(t_i, win_length=512, hop_length=128)
            t_i = torch.istft(t_i, n_fft=512, win_length=512, hop_length=128, normalized=True)

            #t_i = torch.tensor(t_i, requires_grad=True).to(device)
            targets.append(t_i)
            #y_i = librosa.core.istft(y_i, win_length=512, hop_length=128)
            #y_i = torch.tensor(y_i, requires_grad=True).to(device)
            y_i = torch.istft(y_i, n_fft=512, win_length=512, hop_length=128, normalized=True)
            preds.append(y_i)
        else:
            t_i = t[i]
            t_i = t_i[:, :pad_idx]
            y_i = y[i]
            y_i = y_i[:, :pad_idx]
            t_i = torch.istft(t_i, n_fft=512, win_length=512, hop_length=128)
            targets.append(t_i)
            y_i = torch.istft(y_i, n_fft=512, win_length=512, hop_length=128)
            preds.append(y_i)
            x_i = x[i][:, :pad_idx]
            x_i = torch.istft(x_i, n_fft=512, win_length=512, hop_length=128)
            source.append(x_i)
    if x!=None:
        return source, targets, preds
    else:
        return targets, preds


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


def pretrain_critic(clean_path, noisy_path, model_path, num_epochs):

    device = torch.device("cuda:1")
    actor = Actor()
    actor = nn.DataParallel(actor, device_ids=[1, 2])
    actor.load_state_dict(torch.load('/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor/actor_best.pth'))
    actor = actor.to(device)

    critic = Critic()
    critic = critic.to(device)
    critic.apply(init_weights)
    critic = nn.DataParallel(critic, device_ids=[1, 2])

    criterion = CriticLoss()
    criterion.to(device)

    #Training loop
    losses = []
    val_losses = []
    best = copy.deepcopy(critic.state_dict())
    prev_val=99999

    print("Start pretraining...")

    for epoch in range(1, num_epochs+1):
        if epoch <= 100:
            lr = 0.0001
        else:
            lr = lr/100

        optimizer = optim.Adam(critic.parameters(), lr=lr)

        epoch_loss = 0


        dataset = Data(clean_path, noisy_path, 1000)
        loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_custom)
        optimizer = optim.Adam(critic.parameters(), lr=0.001)

        for i, batch in enumerate(loader):
            x = batch["noisy"].unsqueeze(1).to(device)
            t = batch["clean"].unsqueeze(1).to(device)
            m = batch["mask"].to(device)
            out_r, out_i = actor(x)
            out_r = torch.transpose(out_r, 1, 2)
            out_i = torch.transpose(out_i, 1, 2)
            y = predict(x.squeeze(1), (out_r, out_i), floor=True)
            print("predicted", y[:10])
            t = t.squeeze(1)
            x = x.squeeze(1)
            disc_input_y = torch.cat((y, t), 2)
            disc_input_t = torch.cat((t, t), 2)
            disc_input_x = torch.cat((x, t), 2)

            pred_scores = []
            pred_scores.append(critic(disc_input_x))
            pred_scores.append(critic(disc_input_y))
            pred_scores.append(critic(disc_input_t))
            pred_scores = torch.transpose(torch.stack(pred_scores).squeeze(), 0, 1)
            loss = criterion(x, y, t, m, pred_scores, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            epoch_loss+=loss
        
        losses.append(epoch_loss/len(loader))
        np.save(os.path.join(model_path, "loss_critic_pre.npy"), np.array(losses))
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, float(epoch_loss/len(loader))))

        if epoch%5==0:
            ##Validation
            overall_val_loss = 0

            dataset = Data(clean_path, noisy_path, 1000)
            loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_custom)

            for i, batch in enumerate(loader):
                x = batch["noisy"].unsqueeze(1).to(device)
                t = batch["clean"].unsqueeze(1).to(device)
                m = batch["mask"].to(device)
                out_r, out_i = actor(x)
                out_r = torch.transpose(out_r, 1, 2)
                out_i = torch.transpose(out_i, 1, 2)
                y = predict(x.squeeze(1), (out_r, out_i), floor=True)
                t = t.squeeze(1)
                x = x.squeeze(1)
                disc_input_y = torch.cat((y, t), 2)
                disc_input_t = torch.cat((t, t), 2)
                disc_input_x = torch.cat((x, t), 2)

                pred_scores = []
                pred_scores.append(critic(disc_input_x))
                pred_scores.append(critic(disc_input_y))
                pred_scores.append(critic(disc_input_t))
                pred_scores = torch.transpose(torch.stack(pred_scores).squeeze(), 0, 1)
                loss = criterion(x, y, t, m, pred_scores, device)
                overall_val_loss+=loss.detach().cpu().numpy()
                curr_val_loss = overall_val_loss/len(loader)

            val_losses.append(curr_val_loss)
            print('Validation loss: ', curr_val_loss)
            np.save(os.path.join(model_path, 'val_loss_critic_pre.npy'), np.array(val_losses))

            if curr_val_loss < prev_val:
                torch.save(best, os.path.join(model_path, 'critic_best.pth'))
                prev_val = curr_val_loss

            torch.save(best, os.path.join(model_path, "critic_last.pth"))

def pretrain_actor(clean_path, noisy_path, model_path, num_epochs):

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

    print("Start pretraining...")

    for epoch in range(1, num_epochs+1):
        if epoch <= 100:
            lr = 0.0001
        else:
            lr = lr/100

        optimizer = optim.Adam(model.parameters(), lr=lr)

        epoch_loss = 0

        dataset = Data(clean_path, noisy_path, 1000)
        loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_custom)

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
            x = x.squeeze()
            source, targets, preds = inverse(t, y, m, device, x) #Normalization of waveform is made inside inverse()
            loss = criterion(source, targets, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            epoch_loss+=loss
        
        losses.append(epoch_loss/len(loader))
        np.save(os.path.join(model_path, "loss_actor_pre.npy"), np.array(losses))
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))

        if epoch%5==0:
            ##Validation
            overall_val_loss = 0

            dataset = Data(clean_path, noisy_path, 1000)
            loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_custom)

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
                x = x.squeeze()
                source, targets, preds = inverse(t, y, m, device, x)
                loss = criterion(source, targets, preds)
                overall_val_loss+=loss.detach().cpu().numpy()

                curr_val_loss = overall_val_loss/len(loader)
            val_losses.append(curr_val_loss)
            print('Validation loss: ', curr_val_loss)
            np.save(os.path.join(model_path, 'val_loss_actor_pre.npy'), np.array(val_losses))

            if curr_val_loss < prev_val:
                torch.save(best, os.path.join(model_path, 'actor_best.pth'))
                prev_val = curr_val_loss

            torch.save(best, os.path.join(model_path, "actor_last.pth"))


def inference_actor(clean_path, noisy_path, model_path, out_path):
    device = torch.device("cuda")
    model = Actor()
    model = nn.DataParallel(model, device_ids=[0,1])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    fnames = os.listdir(noisy_path)

    print("Num files:", len(fnames))
   
    dataset = DataTest(clean_path, noisy_path)
    loader = data.DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_custom)

    pesq_all = []
    stoi_all = []
    fcount = 0

    for batch in tqdm(loader):
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

        for j in range(len(targets)):
            t_j = targets[j].detach().cpu().numpy()
            p_j = preds[j].detach().cpu().numpy()
            p_j = 10*(p_j/np.linalg.norm(p_j))
            curr_pesq = pesq(t_j, p_j, 16000)
            curr_stoi = stoi(t_j, p_j, 16000)
            pesq_all.append(curr_pesq)
            stoi_all.append(curr_stoi)
            sf.write(os.path.join(out_path, fnames[fcount]) , p_j, 16000)
            fcount+=1

    PESQ = torch.mean(torch.tensor(pesq_all))
    STOI = torch.mean(torch.tensor(stoi_all))

    print("PESQ: ", PESQ, "STOI: ", STOI)

    with open('/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor/test_scores.txt', 'w') as fo:
        fo.write("Avg PESQ: "+str(float(PESQ))+" Avg STOI: "+str(float(STOI)))