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
from data import make_batch

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