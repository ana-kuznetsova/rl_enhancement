import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


from preproc import Data, DataTest
from preproc import collate_custom
from losses import CriticLoss
from modules import Actor, Critic



def train(clean_path, noisy_path, actor_path, critic_path, num_it=100):
    device = torch.device("cuda:1")

    actor = Actor()
    actor = nn.DataParallel(actor, device_ids=[1, 2])
    actor.load_state_dict(torch.load(actor_path))
    actor = actor.to(device)

    critic = Critic()
    critic = nn.DataParallel(critic, device_ids=[1, 2])
    critic.load_state_dict(torch.load(critic_path))
    critic = critic.to(device)
    criterion = CriticLoss()
    criterion.to(device)

    for it in range(1, num_it+1):
        data_actor = Data(clean_path, noisy_path, 200)
        loader_actor = data.DataLoader(data_actor, batch_size=10, shuffle=True, collate_fn=collate_custom)

        data_critic = Data(clean_path, noisy_path, 50)
        loader_critic = data.DataLoader(data_critic, batch_size=5, shuffle=True, collate_fn=collate_custom)

        test_source, test_target = next(iter(loader_actor))
        print(test_source)
    

train('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
      '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
      '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor/actor_best.pth',
      '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_critic/critic_best.pth')