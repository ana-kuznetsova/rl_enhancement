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

pretrain_path = '/N/slate/anakuzne/se_out/pretrain/dnn_l1.pth'

l1 = Layer1()
l1.load_state_dict(torch.load(pretrain_path))

print(l1.fc1.children())

