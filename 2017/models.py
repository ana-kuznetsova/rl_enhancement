import torch
import torch.nn.functional as Func
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import copy
import json

from data import make_batch


class trainDataLoader(data.Dataset):
    def __init__(self, X_chunk, y_chunk):
        self.x = X_chunk
        self.y = y_chunk
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).float(), torch.from_numpy(self.y[index]).float()
    def __len__(self):
        #Number of files
        return self.x.shape[0]


class testDataLoader(data.Dataset):
    def __init__(self, path):
        self.path = path 
        self.test_files = os.listdir(path)
    def __getitem__(self, index):
        xFile = self.path + self.test_files[index]
        X = np.load(xFile)
        return torch.from_numpy(X).t()
    def __len__(self):
        #Number of files
        return len(self.test_files)


class DNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2827, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1339)
        self.drop = nn.Dropout(0.025)
        #self.fc4 = nn.Linear(128, 1339)
        
    def forward(self, x):
        x = Func.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = Func.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x
        

def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)



def train_dnn(num_epochs, model_path, x_path, y_path, 
              loss_path, maxlen=1339, win_len=512, hop_size=256, fs=44000,
              chunk_size=4620, from_pretrained=False):
    model = DNN()
    model.apply(weights)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda")
    model.cuda()
    model = model.to(device)
    criterion.cuda()


    #Training loop

    best_model = copy.deepcopy(model.state_dict())
    best_loss = 9999

    losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        loss = 0.0 
        num_chunk = 4620//chunk_size
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 4620)
            print(start, end)
            X_chunk, y_chunk = make_batch(x_path, y_path, [start, end], 5, maxlen, win_len, hop_size, fs)
            trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 64)
            for step, (audio, target) in enumerate(trainData): 
                audio = audio.to(device)
                target = target.to(device)
                model.train()
                output = model(audio)
                newLoss = criterion(output,target)
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()
            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))
            loss += chunk_loss
        losses.append(loss/num_epochs)
        print('Epoch:{:2},Loss:{:>.5f}'.format(epoch,loss/num_epochs))
    ##Save model, save losses

    with open(loss_path+'losses.json', 'w') as f:
        json.dump(data, f)
    torch.save(best_model, model_path)