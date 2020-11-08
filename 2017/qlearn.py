import os
import numpy as np
import copy
import pickle
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchaudio.transforms import InverseMelScale

from models import weights
from models import DNN_mel
from models import trainDataLoader

from data import window
from metrics import calc_Z
from dnn_rl import DNN_RL
from utils import invert
from utils import read
from dnn_rl import reward
from dnn_rl import R
from dnn_rl import time_weight


def q_learning(num_episodes, x_path, cluster_path, model_path, clean_path, 
               a_path,
               epsilon=0.01, 
               win_len=512,
               hop_size=256,
               fs=16000,
               from_pretrained=False):
    '''
    Params:
        num_episodes: num of updates to the target q function
        x_path: path to the training examples
        y_path: path to the cluster centers
        a_path: path where ground truth actions are stored
        model_path: path where the models are stored
        clean_path: path to clean reference (stft)
    '''
    ### Initialization ###

    P=5 #Window size
    G = np.load(cluster_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 

    ###Load DNN-mapping model
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    
    dnn_map = DNN_mel()
    dnn_map.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))
    dnn_map.cuda()
    dnn_map = dnn_map.to(device)

    #Load MMSE reference Q-function
    q_func_mmse = DNN_RL()
    q_func_mmse.load_state_dict(torch.load(model_path+'qfunc_pretrained.pth'))
    q_func_mmse.cuda()
    q_func_mmse.to(device)
    
    ##Loss
    criterion = nn.MSELoss()
    opt_RMSprop = optim.RMSprop(q_func_mmse.parameters(), lr = 0.001, alpha = 0.9)
    #optimizer = optim.SGD(l1.parameters(), lr=0.01, momentum=0.9)
    criterion.cuda()


    q_losses = []
    reward_sums = []
    
    for ep in range(num_episodes):
        if ep//100:
            print('Episode:{}/{}'.format(ep, num_episodes))

        #Select random
        x_files = os.listdir(x_path)
        x_name = np.random.choice(x_files)

        #phase = pad(np.load(imag_path+x_name), maxlen)

        x_source = np.load(x_path+x_name)
        x = window(x_source, P).T
        x = torch.tensor(x).float().to(device)

        ####### PREDICT DNN-RL AND DNN-MAPPING OUTPUT #######
        Q_pred_mmse = q_func_mmse(x).detach().cpu().numpy() 
        wiener_rl = np.zeros((Q_pred_mmse.shape[0], 64))
        selected_actions = []

        #Predicted actions
        Q_pred_argmax = np.argmax(Q_pred_mmse, axis=1)

        #Load true actions
        true_actions = np.load(a_path + x_name.split('_')[0]+'.npy')

        #Select template index, predict Wiener filter
        for i, action in enumerate(Q_pred_argmax):
        #E-greedy selection for target
            a = np.array([0,1])
            probs = np.array([epsilon, 1-epsilon])
            strategy = np.random.choice(a, p=probs)
            if strategy==0:
                action = np.random.choice(np.arange(32))
            selected_actions.append(int(action))
            G_k_pred = G[action]
            wiener_rl[i] = G_k_pred

        wiener_rl = wiener_rl.T
        y_pred_rl = torch.tensor(np.multiply(x_source, wiener_rl)).float()
        y_pred_dnn = dnn_map(x).T.detach().cpu().numpy()
    
        ##### Calculate reward ######

        x_source_clean = np.load(clean_path+x_name.split('_')[0]+'.npy')
        x_source_wav = invert(x_source_clean)
        y_pred_rl = InverseMelScale(n_stft=257, n_mels=64)(y_pred_rl).detach().cpu().numpy()

        y_pred_dnn_wav =  invert(y_pred_dnn)
        y_pred_rl_wav = invert(y_pred_rl)
        
        z_rl = calc_Z(x_source_wav, y_pred_rl_wav)
        z_map = calc_Z(x_source_wav, y_pred_dnn_wav)
        print('Z-scores:', z_rl, z_map)

        ##PEQS module returns errors
        ##Skip iteration when z-scores are nan
        if np.isnan(z_rl) or np.isnan(z_map):
            print("Skipping iteration...")
            continue

        E = time_weight(y_pred_rl, x_source_clean)
        r = reward(z_rl, z_map, E)
        
        reward_sums.append(np.sum(r))
        np.save(model_path+'reward_sum.npy', np.asarray(reward_sums))

        #### UPDATE Q-FUNCS ####
        Q_func_upd = Q_pred_mmse
        R_cal = R(z_rl, z_map)
        print("R_cal:", R_cal)

        for x_k in range(Q_func_upd.shape[0]):
            a_true = true_actions[x_k]
            a_pred = selected_actions[x_k]

            if a_pred==a_true:
                #print("Pred == True")
                if R_cal > 0:
                    Q_func_upd[x_k][a_pred] = r[a_pred] + np.max(Q_pred_mmse[x_k])
            else:
                #print("R_cal:", R_cal)
                #print("Old:", Q_func_upd[x_k][a_true], "UPD:", Q_pred_mmse[x_k][a_true] - r[x_k])
                if R_cal < 0:
                    Q_func_upd[x_k][a_true] = Q_pred_mmse[x_k][a_true] - r[x_k]
        
        Q_pred_mmse = torch.tensor(Q_pred_mmse)
        Q_func_upd = torch.tensor(Q_func_upd)

        curr_loss = criterion(Q_func_upd, Q_pred_mmse)
        print("Loss:", curr_loss)




'''
        #### UPDATE Q-FUNCS ####

        for i, a_t in enumerate(selected_actions_target):
            a_m = selected_actions_mmse[i]
            if a_t==a_m:
                if R_ > 0:
                    qfunc_target[i][a_t] = r[i] + np.max(qfunc_target[i]) #qfunc shape (1339, 32)
                else:
                    qfunc_target[i][a_t] = Q_pred_rl[i][a_t]
            else:
                if R_ > 0:
                    qfunc_pretrained[i][a_m] = Q_pred_mmse[i][a_m]
                else:
                    qfunc_pretrained[i][a_m] = Q_pred_mmse[i][a_m] - r[i]

        target_tensor = torch.tensor(qfunc_target, requires_grad=True).cuda().float()
        pretrained_tensor = torch.tensor(qfunc_pretrained).cuda().float()
        dnn_rl.train()
        curr_loss = criterion(target_tensor, pretrained_tensor)
        q_losses.append(curr_loss.detach().cpu().numpy())
        np.save(model_path+'q_losses.npy', q_losses)

        print('Episode {}, Training loss:{:>4f}'.format(ep, curr_loss.detach().cpu().numpy()))
        opt_RMSprop.zero_grad()
        curr_loss.backward()
        opt_RMSprop.step()
'''