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


from preproc import Data, DataTest
from preproc import collate_custom
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
        x = x.abs()
        x = 10*torch.log10(x)
        
        #-inf is caused by zero padding
        #Change inf to zeros
        x[x==float("-Inf")] = 0
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        #print(x.shape)
        x = self.conv2d3(x)
        #print(x.shape)
        x = torch.transpose(x.squeeze(), 1, 2)
        x = self.linear1(x)
        x, _ = self.bi_lstm(x)
        x = self.linear2(x)
        #print("X out:", x.shape)
        
        real = x[:,:,:257]
        imag = x[:, :, 257:]
        
        return real, imag
        
    
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


def predict(x, model_out):
    temp = torch.complex(model_out[0], model_out[1])
    return x*temp

def inverse(t, y , m, x):    

    targets = []
    preds = []
    source = []

    for i in range(t.shape[0]):
        pad_idx = int(torch.sum(m[i]))
    
        t_i = t[i]
        t_i = t_i[:, :pad_idx]
        y_i = y[i]
        y_i = y_i[:, :pad_idx]
        #print("Y_inv:", y_i.shape)
        t_i =torch.istft(t_i, n_fft=512, win_length=512, hop_length=128)
        targets.append(t_i)
        y_i = torch.istft(y_i, n_fft=512, win_length=512, hop_length=128)
        print(y_i)
        preds.append(y_i)
        x_i = x[i][:, :pad_idx]
        x_i = torch.istft(x_i, n_fft=512, win_length=512, hop_length=128)
        source.append(x_i)
    return source, targets, preds
    #return targets, preds
    


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


def pretrain_critic(clean_path, noisy_path, model_path, num_epochs):

    device = torch.device("cuda:1")
    actor = Actor()
    actor = nn.DataParallel(actor, device_ids=[1, 2])
    actor.load_state_dict(torch.load('/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor_2/actor_best.pth'))
    actor = actor.to(device)
    actor.eval()

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
            with torch.no_grad():
                out_r, out_i = actor(x)
                out_r = torch.transpose(out_r, 1, 2)
                out_i = torch.transpose(out_i, 1, 2)
            y = predict(x.squeeze(1), (out_r, out_i))
            t = t.squeeze()
            m = m.squeeze()
            x = x.squeeze()
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
            critic.eval()
            for i, batch in enumerate(loader):
                x = batch["noisy"].unsqueeze(1).to(device)
                t = batch["clean"].unsqueeze(1).to(device)
                m = batch["mask"].to(device)
                with torch.no_grad():
                    out_r, out_i = actor(x)
                    out_r = torch.transpose(out_r, 1, 2)
                    out_i = torch.transpose(out_i, 1, 2)
                y = predict(x.squeeze(1), (out_r, out_i))
                t = t.squeeze()
                m = m.squeeze()
                x = x.squeeze()
            
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

            critic.train()
            val_losses.append(curr_val_loss)
            print('Validation loss: ', curr_val_loss)
            np.save(os.path.join(model_path, 'val_loss_critic_pre.npy'), np.array(val_losses))

            if curr_val_loss < prev_val:
                torch.save(best, os.path.join(model_path, 'critic_best.pth'))
                prev_val = curr_val_loss

            torch.save(best, os.path.join(model_path, "critic_last.pth"))

def pretrain_actor(clean_path, noisy_path, model_path, num_epochs):

    device = torch.device("cuda:2")
    model = Actor()
    #model = nn.DataParallel(model, device_ids=[2, 3])
    model = model.to(device)
    model.apply(init_weights)
    

    criterion = SDRLoss()
    criterion.cuda()

    losses = []
    val_losses = []
    best = copy.deepcopy(model.state_dict())
    prev_val=99999

    print("Start pretraining...")

    lr = 0.0001
    for epoch in range(1, num_epochs+1):
        if epoch <= 100:
            lr = 0.0001
        else:
            lr = lr/100

        optimizer = optim.Adam(model.parameters(), lr=lr)

        epoch_loss = 0

        dataset = Data(clean_path, noisy_path, 1000)
        loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_custom)
        model.train()

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
            source, targets, preds = inverse(t, y, m, x)
            loss = criterion(source, targets, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            print("Batch loss:", loss)
            epoch_loss+=loss
        
        losses.append(epoch_loss/len(loader))
        np.save(os.path.join(model_path, "loss_actor_pre.npy"), np.array(losses))
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))

        if epoch%5==0:
            ##Validation
            overall_val_loss = 0

            dataset = Data(clean_path, noisy_path, 1000)
            loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_custom)
            
            model.eval()

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
                source, targets, preds = inverse(t, y, m, x)
                
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
    device = torch.device("cuda:2")
    model = Actor()
    model = nn.DataParallel(model, device_ids=[2, 3])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    fnames = os.listdir(noisy_path)

    print("Num files:", len(fnames))

    pesq_all = []
    stoi_all = []
    fcount = 0

    dataset = DataTest(clean_path, noisy_path)
    loader = data.DataLoader(dataset, batch_size=1, collate_fn=collate_custom)

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
        x = x.squeeze()
        source, targets, preds = inverse(t, y, m, x)

        for j in range(len(targets)):
            t_j = targets[j].detach().cpu().numpy()
            p_j = preds[j].detach().cpu().numpy()
            p_j = 10*(p_j/np.linalg.norm(p_j))
            curr_pesq = pesq(t_j, p_j, 16000)
            curr_stoi = stoi(t_j, p_j, 16000)
            pesq_all.append(curr_pesq)
            stoi_all.append(curr_stoi)
            try:
                sf.write(os.path.join(out_path, fnames[fcount]) , p_j, 16000)
            except IndexError:
                print("Fcount:", fcount, len(fnames))
            fcount+=1

    PESQ = torch.mean(torch.tensor(pesq_all))
    STOI = torch.mean(torch.tensor(stoi_all))

    print("PESQ: ", PESQ, "STOI: ", STOI)

    with open('/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor_1/test_scores.txt', 'w') as fo:
        fo.write("Avg PESQ: "+str(float(PESQ))+" Avg STOI: "+str(float(STOI)))


def enhance(clean_path,noisy_path, model_path, out_path):
    device = torch.device("cuda:2")
    model = Actor()
    model = nn.DataParallel(model, device_ids=[2, 3])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    fnames = os.listdir(noisy_path)
    fcount = 0
    print("FNAMES:", fnames)

    dataset = DataTest(clean_path, noisy_path)
    loader = data.DataLoader(dataset, batch_size=5, collate_fn=collate_custom)

    for batch in tqdm(loader):
        x = batch["noisy"].unsqueeze(1).to(device)
        print("INP:",x.shape)
        m = batch["mask"].to(device)
        out_r, out_i = model(x)
        print("model out:", out_r.shape, out_i.shape)
        out_r = torch.transpose(out_r, 1, 2)
        out_i = torch.transpose(out_i, 1, 2)
        y = predict(x.squeeze(1), (out_r, out_i))

        for i in range(x.shape[0]):
            pad_idx = int(torch.sum(m[i]))
            y_i = y[i]
            y_i = y_i[:, :pad_idx]
            y_i = torch.istft(y_i, n_fft=512, win_length=512, hop_length=128).detach().cpu().numpy()
            y_i = 10*(y_i/np.linalg.norm(y_i))
            sf.write(os.path.join(out_path, "enhanced_"+fnames[fcount]), y_i, 16000)
            fcount+=1
