import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import copy


from preproc import Data, DataTest
from preproc import collate_custom
from losses import CriticLoss, ActorLoss
from modules import Actor, Critic, predict


def update_critic(actor, critic, loader, optimizer, criterion, device):
    epoch_loss=0
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        epoch_loss+=loss
    print()

def update_actor(actor, critic, loader, optimizer, criterion, device):
    epoch_loss = 0
    for i, batch in enumerate(loader):
        x = batch["noisy"].unsqueeze(1).to(device)
        t = batch["clean"].unsqueeze(1).to(device)
        m = batch["mask"].to(device)
        out_r, out_i = actor(x)
        out_r = torch.transpose(out_r, 1, 2)
        out_i = torch.transpose(out_i, 1, 2)
        y = predict(x.squeeze(1), (out_r, out_i), floor=True)
        t = t.squeeze(1)
        disc_input_y = torch.cat((y, t), 2)
        preds = critic(disc_input_y)
        loss = criterion(preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.detach().cpu().numpy()
        epoch_loss+=loss
    print("Actor epoch loss:", epoch_loss/len(loader))


        



def train(clean_path, noisy_path, actor_path, critic_path, num_it=100):
    device = torch.device("cuda:1")

    actor = Actor()
    actor = nn.DataParallel(actor, device_ids=[1, 2])
    actor.load_state_dict(torch.load(actor_path))
    actor = actor.to(device)
    sgd_actor = optim.SGD(actor.parameters(), lr=0.001)
    criterion_actor = ActorLoss()
    criterion_actor.to(device)

    critic = Critic()
    critic = nn.DataParallel(critic, device_ids=[1, 2])
    critic.load_state_dict(torch.load(critic_path))
    critic = critic.to(device)
    sgd_critic = optim.SGD(critic.parameters(), lr=0.001)

    criterion_critic = CriticLoss()
    criterion_critic.to(device)


    critic_losses = []
    val_critic_losses = []
    best_critic = copy.deepcopy(critic.state_dict())
    prev_val_critic=99999

    epoch_loss_critic = 0
    epoch_loss_actor =0

    for it in range(1, num_it+1):
        data_actor = Data(clean_path, noisy_path, 200)
        loader_actor = data.DataLoader(data_actor, batch_size=10, shuffle=True, collate_fn=collate_custom)

        data_critic = Data(clean_path, noisy_path, 50)
        loader_critic = data.DataLoader(data_critic, batch_size=5, shuffle=True, collate_fn=collate_custom)

        update_critic(actor, critic, loader_critic, sgd_critic, criterion_critic, device)
        update_actor(actor, critic, loader_actor, sgd_actor, criterion_actor, device)


        
    

train('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
      '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
      '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor/actor_best.pth',
      '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_critic/critic_best.pth')