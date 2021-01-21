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
from losses import SDRLoss

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
        self.linear1 = nn.Linear(513, 128)
        self.bi_lstm = nn.LSTM(128, hidden_size=256, num_layers=2, 
                               batch_first=True, dropout=0.3, bidirectional=True)
        self.linear2 = nn.Linear(512, 2*513)

    def forward(self, x):
        x = x.real
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = x.squeeze(1)
        
        x_batch = []

        ##Run through linear layer
        for i in range(x.shape[0]):
            curr_x = self.linear1(x[i].T)
            x_batch.append(curr_x)
        x = torch.stack(x_batch)
        del x_batch
    
        x, (h, _) = self.bi_lstm(x)

        x_batch = []
        for i in range(x.shape[0]):
            curr_x = self.linear2(x[i])
            x_batch.append(curr_x)

        real = []
        imag = []
        
        for m in x_batch:
            r = m[:,:int(m.shape[1]/2)]
            real.append(r)
            i = m[:,int(m.shape[1]/2):]
            imag.append(i)
        del x_batch

        return torch.stack(real), torch.stack(imag)
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.bnorm = nn.BatchNorm1d(513)
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
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(50*4*37, 50)
        self.fc2 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = self.bnorm(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = self.conv2d4(x)
        x = self.avg_pool(x)
        x = self.fc1(self.flat(x))
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

def inverse(t, y , m):
    targets = []
    preds = []

    for i in range(t.shape[0]):
        pad_idx = int(torch.sum(m[i]))
        t_i = t[i].squeeze(0)
        t_i = t_i[:, :pad_idx]
        y_i = y[i].squeeze(0)
        y_i = y_i[:, :pad_idx]
        t_i = torch.istft(t_i, n_fft=1024, win_length=512, hop_length=128)
        targets.append(t_i)
        y_i = torch.istft(y_i, n_fft=1024, win_length=512, hop_length=128)
        preds.append(y_i)
    return targets, preds

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

def pretrain_critic():

    device = torch.device("cuda:1")
    actor = Actor()
    actor.load_state_dict(torch.load('/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor/actor_best.pth'))
    actor = actor.to(device)

    critic = Critic()
    critic = critic.to(device)

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
        t = t.unsqueeze(1)
        disc_input = torch.cat((y, t), 2)
        print(disc_input.shape)

def pretrain_actor(clean_path, noisy_path, model_path, num_epochs):

    device = torch.device("cuda:1")
    model = Actor()
    model.cuda()
    model = model.to(device)
    model.apply(init_weights)

    criterion = SDRLoss()
    criterion.cuda()

    losses = []
    val_losses = []
    best = copy.deepcopy(model.state_dict())
    prev_val=99999

    for epoch in range(1, num_epochs+1):
        if epoch <= 100:
            lr = 0.0001
        else:
            lr = lr/100

        optimizer = optim.Adam(model.parameters(), lr=lr)

        epoch_loss = 0

        dataset = DataLoader(clean_path, noisy_path, get_feats, 1000)
        loader = data.DataLoader(dataset, batch_size=5, shuffle=True)

        for batch in loader:
            x = batch["noisy"].unsqueeze(1).to(device)
            t = batch["clean"].unsqueeze(1).to(device)
            m = batch["mask"].to(device)
            out_r, out_i = model(x)
            out_r = torch.transpose(out_r, 1, 2)
            out_i = torch.transpose(out_i, 1, 2)
            y = predict(x.squeeze(1), (out_r, out_i))
            targets, preds = inverse(t, y, m)
            loss = criterion(targets, preds)
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

            dataset = DataLoader(clean_path, noisy_path, get_feats, 500)
            loader = data.DataLoader(dataset, batch_size=5, shuffle=True)

            for batch in loader:
                x = batch["noisy"].unsqueeze(1).to(device)
                t = batch["clean"].unsqueeze(1).to(device)
                m = batch["mask"].to(device)
                out_r, out_i = model(x)
                out_r = torch.transpose(out_r, 1, 2)
                out_i = torch.transpose(out_i, 1, 2)
                y = predict(x.squeeze(1), (out_r, out_i))
                targets, preds = inverse(t, y, m)
                loss = criterion(targets, preds)
                overall_val_loss+=loss.detach().cpu().numpy()

                curr_val_loss = overall_val_loss/len(loader)
            val_losses.append(curr_val_loss)
            print('Validation loss: ', curr_val_loss)
            np.save(os.path.join(model_path, 'val_loss_actor_pre.npy'), np.array(val_losses))

            if curr_val_loss < prev_val:
                torch.save(best, os.path.join(model_path, 'actor_best.pth'))
                prev_val = curr_val_loss

            torch.save(best, os.path.join(model_path, "actor_last.pth"))


def inference(clean_path, noisy_path, model_path):
    device = torch.device("cuda:1")
    model = Actor()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
   
    dataset = DataLoader(clean_path, noisy_path, get_feats, 2)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(loader):
        x = batch["noisy"].unsqueeze(1).to(device)
        t = batch["clean"].unsqueeze(1).to(device)
        m = batch["mask"].to(device)
        out_r, out_i = model(x)
        out_r = torch.transpose(out_r, 1, 2)
        out_i = torch.transpose(out_i, 1, 2)
        y = predict(x.squeeze(1), (out_r, out_i))
        targets, preds = inverse(t, y, m)
        
        sf.write('target_'+str(i)+'.wav', targets[0].detach().cpu().numpy(), 16000)
        sf.write('pred_'+str(i)+'.wav', preds[0].detach().cpu().numpy(), 16000)