import os
import numpy as np
import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from models import weights
from models import DNN_mel
from data import mel_spec
from data import pad
from data import get_X_batch
from utils import invert
from metrics import calc_Z

#### REWARD DEFINITION ####

def reward(z_scores, E):
    '''
    z_scores[0]: predicted G from DNN-RL
    z_scores[1]: predicted G from DNN-map
    '''
    R_ = R(z_scores)
    if R_ > 0:
        return (1 - E)*R_
    else:
        return E*R_


def R(z_scores, alpha=20):
    return np.tanh(alpha(preds[0]- preds[1]))

def time_weight(Y, S):
    '''
    E - calculated time weight
    Y: predicted spectrogram
    S: true spectrogram
    '''
    E_approx = np.sum(np.abs(np.log(Y) - np.log(S))**2, axis=0)
    E = E_approx/np.max(E_approx)
    return E



class DNN_RL(nn.Module):
    def __init__(self, l1_2=None):
        super().__init__()
        self.fc1 = nn.Linear(704, 64)
        self.fc2 = nn.Linear(64, 32)
        self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.soft(x)
        return x 


def q_learning(x_path, y_path, model_path, clean_path,
               imag_path='/nobackup/anakuzne/data/snr0_train_img/',
               num_episodes=50000, epsilon=0.01, maxlen=1339, 
               win_len=512,
               hop_size=256,
               fs=16000,
               from_pretrained=False):
    '''
    Params:
        x_path: path to the training examples
        y_path: path to the cluster centers
        model_path: path to dir where DNN-mapping model is stored
        clean_path: path to clean reference (stft)
    '''
    ### Initialization ###
    P=5 #Window size
    G = np.load(y_path) #Cluster centers for wiener masks

    ###Load DNN-mapping model
    
    dnn_map = DNN_mel()
    dnn_map.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))
    dnn_map = dnn_map.to("cuda")
    
    
    dnn_rl = DNN_RL()
    dnn_rl.apply(weights)
    dnn_rl = dnn_rl.to("cuda")

    #criterion = nn.MSELoss()
    #optimizer = optim.SGD(dnn_rl.parameters(), lr=0.01, momentum=0.9)
    #device = torch.device('cuda:3')
    #device2 = torch.device('cuda:1')
    #dnn_rl.cuda()
    #dnn_rl = dnn_rl.to(device)
    #criterion.cuda()

    #Select random
    x_files = os.listdir(x_path)
    x_name = np.random.choice(x_files)

    phase = pad(np.load(imag_path+x_name), maxlen)

    x_source = np.load(x_path+x_name)
    x = mel_spec(x_source, win_len, hop_size, fs)
    x = np.abs(get_X_batch(x, P)).T
    x = pad(x, maxlen).T
    x = torch.tensor(x).float().cuda()
    print('X:', x.size())
    '''
    ####### PREDICT DNN-RL AND DNN-MAPPING OUTPUT #######
    rl_out = dnn_rl(x)
    wiener_rl = np.zeros((1339, 257))

    print("OUT:", rl_out)
    
    #Select template index, predict Wiener filter
    for i, row in enumerate(rl_out):
        #ind = np.argmax(row.detach().numpy())
        print(row.detach().cpu().numpy())
        #ind = row.cpu()
        #print('IND:', ind)
        G_k_pred = G[ind]
        wiener_pred[i] = G_k_pred

    wiener_rl = wiener_rl.T
    y_pred_rl = np.multiply(pad(x_source, maxlen), wiener_rl) + phase  

    map_out = dnn_map(x)
    wiener_map = map_out.detach().numpy()
    y_pred_map = np.multiply(pad(x_source, maxlen), wiener_map) + phase  

    ##### Calculate reward ######
    x_source_wav = invert(x_source)
    y_map_wav = invert(y_pred_map)[:x_source_wav.shape[0]]
    y_rl_wav = invert(y_pred_map)[:x_source_wav.shape[0]]

    z_rl = calc_Z(x_source_wav, y_rl_wav)
    z_map = calc_Z(x_source_wav, y_map_wav)
    #print('Z-scores:', z_rl, z_map)
    
    clean = np.load(clean_path+x_name)
    E = time_weight(y_pred_rl, pad(clean, maxlen))
    r = reward([z_rl, z_map], E)
    '''