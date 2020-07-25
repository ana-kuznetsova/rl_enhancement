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

newmodel = torch.nn.Sequential(*(list(l1.children())[:-2]))

l2 = Layer_1_2(newmodel)
criterion = nn.MSELoss()
optimizer = optim.SGD(l2.parameters(), lr=0.01, momentum=0.9)
device = torch.device("cuda")
l2.cuda()
l2 = l2.to(device)
criterion.cuda()
best_l2 = copy.deepcopy(l2.state_dict())
