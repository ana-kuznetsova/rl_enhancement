import torch
import torch.nn.functional as Func
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import copy
import pickle
import pandas as pd
import os

from data import make_batch
from data import make_batch_test
from data import pad
from metrics import eval_pesq


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
    def __init__(self, X_chunk):
        self.x = X_chunk
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).float()
    def __len__(self):
        #Number of files
        return self.x.shape[0]



class DNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2827, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 257)
        self.drop = nn.Dropout(0.025)
        #self.fc4 = nn.Linear(128, 1339)
        
    def forward(self, x):
        #x = Func.sigmoid(self.fc1(x))
        x = Func.relu(self.fc1(x))
        x = self.drop(x)
        #x = Func.sigmoid(self.fc2(x))
        x = Func.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x        

class DNN_pretrain(nn.Module):
    def __init__(self, layer):
        super().__init__()
        
        if layer==1:
            self.fc1 = nn.Linear(2827, 128)
            self.fc2 = nn.Linear(128, 1339)
        elif layer==2:
            self.fc1 = nn.Linear(2827, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 1339)
        self.drop = nn.Dropout(0.025)

        
    def forward(self, x, layer):
        if layer==1:
            x = Func.sigmoid(self.fc1(x))
            x = self.drop(x)
            x = Func.sigmoid(self.fc2(x))
            return x
        elif layer==2:
            x = Func.sigmoid(self.fc1(x))
            x = self.drop(x)
            x = Func.sigmoid(self.fc2(x))
            x = self.drop(x)
            x = self.drop(x)
            x = self.fc3(x)
            return x
        

def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)



def train_dnn(num_epochs, model_path, x_path, y_path, 
              loss_path, chunk_size,
              maxlen=1339, win_len=512, hop_size=256, fs=44000, from_pretrained=False):
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
        
        num_chunk = (4620//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 4620)
            print(start, end)

            X_chunk, y_chunk = make_batch(x_path, y_path, [start, end], 5, maxlen, win_len, hop_size, fs)
            trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 128)

            for step, (audio, target) in enumerate(trainData): 
                #print('Step:', step)
                audio = audio.to(device)
                target = target.to(device)
                model.train()
                output = model(audio)
                newLoss = criterion(output,target)
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()
            
            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss/num_chunk))

        loss += chunk_loss/num_chunk

        losses.append(loss/num_epochs)
        print('Epoch:{:2},Loss:{:>.5f}'.format(epoch,loss/epoch))
    ##Save model, save losses

    pickle.dump(losses, open( loss_path+"losses.p", "wb" ) )

    torch.save(best_model, model_path+'dnn_map_best.pth')



def pretrain(num_epochs, model_path, x_path, y_path, weights_path,
              loss_path, maxlen=1339, win_len=512, hop_size=256, fs=44000,
              chunk_size=4620):
    
    min_delta = 0.05

    #Pretrain first layer

    pass


def inference(test_data_path, clean_test_path, out_test, model_path, imag_path, chunk_size, maxlen=1339,
             win_len=512, hop_size=256, fs=44000):
    model = DNN()
    model.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))
    fnames = os.listdir(test_data_path)

    num_chunk = (1680//chunk_size) +1
    for chunk in range(num_chunk):
        chunk_loss = 0
        start = chunk*chunk_size
        end = min(start+chunk_size, 1680)
        print(start, end)
        x_list = [test_data_path + n for n in fnames]
        X_chunk = make_batch_test(x_list, [start, end], 5, maxlen, win_len, hop_size, fs)
        testData = data.DataLoader(testDataLoader(X_chunk), batch_size = 1339)

        chunk_names = fnames[start:end]
        for step, audio in enumerate(testData):
            #print('Step:', step)

            name = chunk_names[step]
            with torch.no_grad():
                output = model(audio)
                output = np.transpose(output.cpu().data.numpy().squeeze())
                ##Restore phase (imaginary part)
                imag = pad(np.load(imag_path+name), maxlen)
                np.save(out_test+name, np.exp(output)+imag)
            #output = librosa.istft(np.transpose(output[0].cpu().data.numpy().squeeze()), hop_length=hop_size,
            #                    win_length=win_len) 
            #    librosa.output.write_wav(out_test+name, output, fs) 

    #eval_pesq(out_test, clean_test_path, out_test)
