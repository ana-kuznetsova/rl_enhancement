import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import copy
import os
import numpy as np
from pystoi import stoi
from pypesq import pesq


from preproc import Data, DataTest
from preproc import collate_custom
from losses import CriticLoss, ActorLoss
from modules import Actor, Critic, predict, inverse


def update_critic(actor, critic, loader, optimizer, criterion, device):
    epoch_loss=0
    for batch in loader:
        x = batch["noisy"].unsqueeze(1).to(device)
        t = batch["clean"].unsqueeze(1).to(device)
        m = batch["mask"].to(device)
        actor.eval()
       
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
        actor.train()
    return epoch_loss/len(loader)

def update_actor(actor, critic, loader, optimizer, criterion, device):
    epoch_loss = 0
    critic.eval()
    for batch in loader:
        x = batch["noisy"].unsqueeze(1).to(device)
        t = batch["clean"].unsqueeze(1).to(device)
        m = batch["mask"].to(device)
        out_r, out_i = actor(x)
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
        epoch_loss+=loss
    critic.train()
    return epoch_loss/len(loader)


def calc_metrics(loader, actor, device):
    pesq_all = []
    stoi_all = []
    for batch in loader:
        x = batch["noisy"].unsqueeze(1).to(device)
        t = batch["clean"].unsqueeze(1).to(device)
        m = batch["mask"].to(device)
        out_r, out_i = actor(x)
        out_r = torch.transpose(out_r, 1, 2)
        out_i = torch.transpose(out_i, 1, 2)
        y = predict(x.squeeze(1), (out_r, out_i))
        t = t.squeeze()
        m = m.squeeze()
        source, targets, preds = inverse(t, y, m, x)

        for j in range(len(targets)):
            curr_pesq = pesq(targets[j].detach().cpu().numpy(), preds[j].detach().cpu().numpy(), 16000)
            curr_stoi = stoi(targets[j].detach().cpu().numpy(), preds[j].detach().cpu().numpy(), 16000)
            pesq_all.append(curr_pesq)
            stoi_all.append(curr_stoi)
    PESQ = torch.mean(torch.tensor(pesq_all))
    STOI = torch.mean(torch.tensor(stoi_all))
    return PESQ, STOI



def train(clean_path, noisy_path, clean_test, noisy_test, actor_path, critic_path, model_path, num_it=100):
    device = torch.device("cuda:1")

    actor = Actor()
    actor = nn.DataParallel(actor, device_ids=[1, 2])
    actor.load_state_dict(torch.load(actor_path))
    actor = actor.to(device)
    sgd_actor = optim.SGD(actor.parameters(), lr=0.001)
    criterion_actor = ActorLoss()
    criterion_actor.to(device)

    critic = Critic()
    critic = nn.DataParallel(critic, device_ids=[3, 0])
    critic.load_state_dict(torch.load(critic_path))
    critic = critic.to(device)
    sgd_critic = optim.SGD(critic.parameters(), lr=0.001)

    criterion_critic = CriticLoss()
    criterion_critic.to(device)


    critic_losses = []
    best_critic = copy.deepcopy(critic.state_dict())

    actor_losses = []
    best_actor = copy.deepcopy(critic.state_dict())

    prev_actor_loss = 999999.0
    prev_critic_loss = 999999.0

    pesq_all = []
    stoi_all = []


    for it in range(1, num_it+1):
        data_actor = Data(clean_path, noisy_path, 200)
        loader_actor = data.DataLoader(data_actor, batch_size=10, shuffle=True, collate_fn=collate_custom)

        data_critic = Data(clean_path, noisy_path, 50)
        loader_critic = data.DataLoader(data_critic, batch_size=5, shuffle=True, collate_fn=collate_custom)

        epoch_loss_critic = update_critic(actor, critic, loader_critic, sgd_critic, criterion_critic, device)
        critic_losses.append(epoch_loss_critic)
        epoch_loss_actor = update_actor(actor, critic, loader_actor, sgd_actor, criterion_actor, device)
        actor_losses.append(epoch_loss_actor)

        print('Epoch:{:2} Actor loss:{:>4f} Critic loss:{:>4f}'.format(it, float(epoch_loss_actor), float(epoch_loss_critic)))

        ### Save models

        if epoch_loss_actor < prev_actor_loss:
            torch.save(best_actor, os.path.join(model_path, 'actor_best.pth'))
            prev_actor_loss = epoch_loss_actor
        
        if epoch_loss_critic < prev_critic_loss:
            torch.save(best_critic, os.path.join(model_path, 'critic_best.pth'))
            prev_critic_loss = epoch_loss_critic

        np.save(os.path.join(model_path, 'actor_loss.npy'), np.array(actor_losses))
        np.save(os.path.join(model_path, 'critic_loss.npy'), np.array(critic_losses))

        ###Save models every 5 it for plotting weight landscape
        if it%5==0:
            torch.save(best_actor, os.path.join(model_path, "weights", "actor_"+str(it)+".pth"))
            torch.save(best_critic, os.path.join(model_path, "weights", "critic_"+str(it)+".pth"))

        ### PESQ of predictions
        data_test = DataTest(clean_test, noisy_test)
        loader_test = data.DataLoader(data_test, batch_size=5, shuffle=False, collate_fn=collate_custom)

        PESQ, STOI = calc_metrics(loader_test, actor, device)
        print("PESQ:", PESQ, "STOI:", STOI)

        pesq_all.append(PESQ)
        stoi_all.append(STOI)
        np.save(os.path.join(model_path, "pesq.npy"), np.array(pesq_all))
        np.save(os.path.join(model_path, "stoi.npy"), np.array(stoi_all))

        

train(clean_path='/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
      noisy_path='/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/',
      clean_test='/nobackup/anakuzne/data/voicebank-demand/clean_testset_wav/',
      noisy_test='/nobackup/anakuzne/data/voicebank-demand/noisy_testset_wav/',
      actor_path='/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor_1/actor_best.pth',
      critic_path='/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_critic_1/critic_best.pth',
      model_path='/nobackup/anakuzne/data/experiments/speech_enhancement/2020/actor_critic_1/')