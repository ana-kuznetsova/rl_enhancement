import os
import numpy as np
import copy
import pickle
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from models import weights
from models import DNN_mel

from metrics import calc_Z
from dnn_rl import DNN_RL
from dnn_rl import reward
from dnn_rl import R
from dnn_rl import time_weight
from preproc import q_transform
from preproc import invert_mel
from preproc import invert


def q_learning(x_path, noise_path, cluster_path, model_path,
               num_episodes=10000,
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
        model_path: path where the pretrained models are stored
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
    
    curr_qfunc = copy.deepcopy(q_func_mmse.state_dict())

    ##Loss
    criterion = nn.MSELoss()
    opt_RMSprop = optim.RMSprop(q_func_mmse.parameters(), lr = 0.001, alpha = 0.9)
    criterion.cuda()


    q_losses = []
    reward_sums = []
    curr_losses = []
    
    for ep in range(num_episodes):

        #Select random
        x_files = os.listdir(x_path)
        x_name = np.random.choice(x_files)

        sample = q_transform(os.path.join(x_path, x_name), noise_path, cluster_path, 0, 5)
        x = torch.tensor(sample['x']).to(device)
        true_actions = sample['t']

        ####### PREDICT DNN-RL AND DNN-MAPPING OUTPUT #######
        Q_pred_mmse = q_func_mmse(x).detach().cpu().numpy() 
        wiener_rl = np.zeros((Q_pred_mmse.shape[0], 64))
        selected_actions = []

        #Predicted actions
        Q_pred_argmax = np.argmax(Q_pred_mmse, axis=1)

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
        y_pred_rl = torch.tensor(np.multiply(x, wiener_rl)).float()
        y_pred_dnn = dnn_map(x).T.detach().cpu().numpy()
    
        ##### Calculate reward ######

        x_clean = librosa.core.load(os.path.join(x_path, x_name), 16000, mono=True)
        y_pred_rl = invert_mel(y_pred_rl)

        y_pred_dnn_wav =  invert(y_pred_dnn)
        y_pred_rl_wav = invert(y_pred_rl)
        
        z_rl = calc_Z(x_clean, y_pred_rl_wav)
        z_map = calc_Z(x_clean, y_pred_dnn_wav)
        #print('Z-scores:', z_rl, z_map)

        ##PEQS module returns errors
        ##Skip iteration when z-scores are nan
        if np.isnan(z_rl) or np.isnan(z_map):
            print("Skipping iteration...")
            continue

        E = time_weight(y_pred_rl, x_clean)
        r = reward(z_rl, z_map, E)

        #### UPDATE Q-FUNCS ####
        Q_func_upd = Q_pred_mmse
        R_cal = R(z_rl, z_map)
        #print("R_cal:", R_cal)

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

        # Normalize updated Q-func

        Q_func_upd =  Q_func_upd/Q_func_upd.sum(axis=1, keepdims=True)
        
        Q_pred_mmse = torch.tensor(Q_pred_mmse, requires_grad=True)
        Q_func_upd = torch.tensor(Q_func_upd, requires_grad=True)

        q_func_mmse.train()
        curr_loss = criterion(Q_func_upd, Q_pred_mmse)
        curr_losses.append(curr_loss.detach().cpu().numpy())

        if ep%100 == 0:
            ## Save losses
            avg_loss = np.sum(np.array(curr_losses)/len(curr_losses))
            q_losses.append(avg_loss)
            curr_losses = []
            np.save(model_path+'q_losses.npy', q_losses)
            print('Episode {}, Training loss:{:>4f}'.format(ep, avg_loss))

            ## Save rewards
            reward_sums.append(np.sum(r))
            np.save(model_path+'reward_sum.npy', np.array(reward_sums))

            ## Save model
            torch.save(curr_qfunc, model_path+'qfunc_model.pth')

        opt_RMSprop.zero_grad()
        curr_loss.backward()
        opt_RMSprop.step()


def qlean_predict(model_path, x_path, out_path):
    torch.cuda.empty_cache() 
    device = torch.device('cuda')

    q_func_pretrained = DNN_RL()
    q_func_pretrained.load_state_dict(torch.load(model_path+'qfunc_model.pth'))
    q_func_pretrained.cuda()