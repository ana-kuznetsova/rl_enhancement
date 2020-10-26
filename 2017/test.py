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
from dnn_rl import eval_actions
from data import create_noisy_data
from models import pretrain


eval_actions(model_path='/N/slate/anakuzne/se_data/qfunc_pretrain/',
             x_path="/N/slate/anakuzne/se_data/snr0_train_melspecs/",
             a_path='/N/slate/anakuzne/se_data/action_labels/')

'''
MMSE_train(chunk_size=1000, x_path='/nobackup/anakuzne/data/snr0_train_melspecs/', 
           a_path='/nobackup/anakuzne/data/action_labels/',
           model_path='/nobackup/anakuzne/data/qfunc_pretrain/')

pretrain(chunk_size=500, x_path='/N/slate/anakuzne/se_data/snr0_train_melspecs/',
        y_path='/N/slate/anakuzne/se_data/snr0_train_clean/', 
        model_path='/N/slate/anakuzne/se_data/model_log_amp/', 
        loss_path='/N/slate/anakuzne/se_data/model_log_amp/')

MMSE_train(chunk_size=1000, x_path='/N/slate/anakuzne/se_data/snr0_train/', 
           y_path='/N/slate/anakuzne/se_data/snr0_w_target/',
           a_path='/N/slate/anakuzne/se_data/action_labels/',
           model_path='/N/slate/anakuzne/se_data/qfunc_pretrain/',
           cluster_path = '/N/slate/anakuzne/se_data/kmeans_centers.npy',
           clean_path = '/N/slate/anakuzne/se_data/snr0_train_clean/',
           imag_path= '/N/slate/anakuzne/se_data/snr0_train_img/')


create_noisy_data(x_path='/N/project/aspire_research_cs/Data/Corpora/Speech/TIMIT/corpus/',
                out_path='/N/slate/anakuzne/se_data/snr036_mel_train/',
                noise_path='/N/project/aspire_research_cs/Data/Corpora/Noise/cafe_16k.wav')
'''