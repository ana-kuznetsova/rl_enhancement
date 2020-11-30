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



pretrain('/nobackup/anakuzne/data/koizumi17_timit/TIMIT/corpus/', 
        '/nobackup/anakuzne/data/koizumi17_timit/config_test/', 100,
        '/nobackup/anakuzne/data/koizumi17_timit/cafe_16k.wav', 0, 5)