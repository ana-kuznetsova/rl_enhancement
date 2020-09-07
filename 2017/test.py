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

'''
pretrain_path = '/u/anakuzne/data/pretrain_sig/dnn_l1.pth'

l1 = Layer1()
l1.load_state_dict(torch.load(pretrain_path))


l2 = Layer_1_2()
l2.load_state_dict(torch.load('/u/anakuzne/data/pretrain_sig/dnn_l2.pth'))
#print('L2:', l2)

#print('Weights L1:', l2.fc1.weight)


dnn = DNN_mel(l2)
print('DNN:', dnn)
print('Weights L1:', dnn.fc1.weight)
print('Weights L2:', dnn.fc2.weight)


criterion = nn.MSELoss()
optimizer = optim.SGD(l2.parameters(), lr=0.01, momentum=0.9)
device = torch.device("cuda")
l2.cuda()
l2 = l2.to(device)
criterion.cuda()
best_l2 = copy.deepcopy(l2.state_dict())
'''
'''
q_learning(x_path='/nobackup/anakuzne/data/snr0_train/', 
           y_path='/nobackup/anakuzne/data/kmeans_centers.npy', 
           model_path='/nobackup/anakuzne/data/model_wiener_50/',
           clean_path='/nobackup/anakuzne/data/snr0_train_clean/')
'''


'''
MMSE_pretrain(chunk_size=1000, x_path='/nobackup/anakuzne/data/snr0_train/', 
           y_path='/nobackup/anakuzne/data/snr0_w_target/', 
           model_path='/nobackup/anakuzne/data/qfunc_pretrain/',
           cluster_path = '/nobackup/anakuzne/data/kmeans_centers.npy',
           clean_path = '/nobackup/anakuzne/data/snr0_train_clean/')
MMSE_train(chunk_size=1000, x_path='/nobackup/anakuzne/data/snr0_train/', 
           y_path='/nobackup/anakuzne/data/snr0_w_target/', 
           model_path='/nobackup/anakuzne/data/qfunc_pretrain/',
           cluster_path = '/nobackup/anakuzne/data/kmeans_centers.npy',
           clean_path = '/nobackup/anakuzne/data/snr0_train_clean/')
'''

MMSE_pretrain(chunk_size=1000, x_path='/N/slate/anakuzne/se_data/snr0_train/', 
           y_path='/N/slate/anakuzne/se_data/snr0_w_target/',
           a_path='/N/slate/anakuzne/se_data/action_labels/'
           model_path='/N/slate/anakuzne/se_data/qfunc_pretrain/',
           cluster_path = '/N/slate/anakuzne/se_data/kmeans_centers.npy',
           clean_path = '/N/slate/anakuzne/se_data/snr0_train_clean/',
           imag_path= '/N/slate/anakuzne/se_data/snr0_train_img/')