import torch
import torch.nn.functional as Func
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import copy

from models import Layer1
from models import Layer_1_2
from models import DNN_mel

#from dnn_rl import q_learning
from dnn_rl import MMSE_pretrain
from dnn_rl import MMSE_train


MMSE_pretrain(chunk_size=100, x_path='/N/slate/anakuzne/se_data/snr0_train/', 
           y_path='/N/slate/anakuzne/se_data/snr0_w_target/',
           a_path='/N/slate/anakuzne/se_data/action_labels/',
           model_path='/N/slate/anakuzne/se_data/qfunc_pretrain/',
           cluster_path = '/N/slate/anakuzne/se_data/kmeans_centers.npy',
           clean_path = '/N/slate/anakuzne/se_data/snr0_train_clean/',
           imag_path= '/N/slate/anakuzne/se_data/snr0_train_img/')
'''
MMSE_train(chunk_size=1000, x_path='/N/slate/anakuzne/se_data/snr0_train/', 
           y_path='/N/slate/anakuzne/se_data/snr0_w_target/',
           a_path='/N/slate/anakuzne/se_data/action_labels/',
           model_path='/N/slate/anakuzne/se_data/qfunc_pretrain/',
           cluster_path = '/N/slate/anakuzne/se_data/kmeans_centers.npy',
           clean_path = '/N/slate/anakuzne/se_data/snr0_train_clean/',
           imag_path= '/N/slate/anakuzne/se_data/snr0_train_img/')

'''