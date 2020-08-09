import os

import torch
import torch.nn.functional as Func
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import copy
import pickle

import numpy as np
from models import trainDataLoader
from models import testDataLoader
from models import weights
from data import mel_spec
from data import pad
from data import get_X_batch

#### REWARD DEFINITION ####

def reward(preds, E):
    R_ = R(preds)
    if R_ > 0:
        return (1 - E)*R_
    else:
        return E*R_


def R(preds, alpha=20):
    '''
    Preds[0]: predicted G from DNN-RL
    Preds[1]: predicted G from DNN-map
    '''
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
        x = Func.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = Func.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.soft(x)
        return x 


def q_learning(x_path, y_path, 
               num_episodes=50000, epsilon=0.01, maxlen=1339, 
               win_len=512,
               hop_size=256,
               fs=16000,
               from_pretrained=False):
    '''
    Params:
        x_path: path to the training examples
        y_path: path to the cluster centers
    '''
    ### Initialization ###
    P=5 #Window size

    dnn_rl = DNN_RL()
    dnn_rl.apply(weights)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(dnn_rl.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda")
    dnn_rl.cuda()
    dnn_rl = dnn_rl.to(device)
    criterion.cuda()

    ## Make training example, put into data loader
    X = []
    Y = []

    #Select random
    y = np.load(y_path).T
    x_files = os.listdir(x_path)
    x = np.random.choice(x_files)

    x = np.load(x_path+x)
    x = mel_spec(x, win_len, hop_size, fs)
    x = np.abs(get_X_batch(x, P)).T
    x = pad(x, maxlen)

    for col in range(x.shape[1]):
        x_k = np.repeat(x[:, col].reshape(-1, 1), y.shape[1], axis=1)
        X.append(x_k)
        Y.append(y)
    #Input     
    X = np.asarray(X) #(1339, 704, 32)
    Y = np.asarray(Y) #(1339, 257, 32)

    trainData = data.DataLoader(trainDataLoader(X, Y), batch_size = 1)
    
    for step, (audio, target) in enumerate(trainData): 
        audio = audio.to(device)
        target = target.to(device)
        dnn_rl.train()
        output = dnn_rl(audio)