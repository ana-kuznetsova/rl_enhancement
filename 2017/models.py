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




class DNN_mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(704, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 257)
        self.drop = nn.Dropout(0.3)
    
    def forward(self, x):
        x = Func.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = Func.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x
        
class Layer1(nn.Module):
    '''
    Train with mel features
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(704, 128)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 257)

    def forward(self, x):
        x = Func.relu(self.fc1(x))
        x = self.drop(x)
        return self.out(x)

class Layer_1_2(nn.Module):
    def __init__(self, l1):
        super().__init__()
        self.fc1 = l1
        self.fc2 = nn.Linear(128, 128)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 257)

    def forward(self, x):
        x = Func.relu(self.fc1(x))
        x = Func.relu(self.fc2(x))
        x = self.drop(x)
        return self.out(x)

class DNN(nn.Module):
    def __init__(self, l1_2=None):
        super().__init__()
        if l1_2:
            self.fc1 = l1_2.fc1
            self.fc2 = l1_2.fc2
        else:
            self.fc1 = nn.Linear(2827, 128)
            self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 257)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, x):
        #x = Func.sigmoid(self.fc1(x))
        x = Func.relu(self.fc1(x))
        x = self.drop(x)
        #x = Func.sigmoid(self.fc2(x))
        x = Func.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x 
        

def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)


def pretrain(chunk_size, model_path, x_path, y_path, loss_path, num_epochs=1,
            maxlen=1339, win_len=512, hop_size=256, fs=16000):
    
    #temp change later
    feat_type='mel'

    min_delta = 0.05 #Min change in loss which can be considered as improvement
    stop_epoch = 15 #Number of epochs without improvement
    no_improv = 0
    prev_loss = 1

    losses_l1 = []
    losses_l2 = []

    ############# PRETRAIN FIRST LAYER ################
    '''
    l1 = Layer1()
    l1.apply(weights)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(l1.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda")
    l1.cuda()
    l1 = l1.to(device)
    criterion.cuda()

    best_l1 = copy.deepcopy(l1.state_dict())

    print("Start PRETRAINING first layer...")

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))

        epoch_loss = 0.0

        num_chunk = (4620//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 4620)
            print(start, end)

            X_chunk, y_chunk = make_batch(x_path, y_path, 
                                         [start, end], 5, 
                                         maxlen, win_len, 
                                         hop_size, feat_type, fs)

            trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 128)

            for step, (audio, target) in enumerate(trainData): 
                audio = audio.to(device)
                target = target.to(device)
                l1.train()
                output = l1(audio)
                newLoss = criterion(output,target)                
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()
            
            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(trainData)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        #Check for early stopping
        losses_l1.append(epoch_loss/num_chunk)
        pickle.dump(losses_l1, open(loss_path+"losses_l1.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))

        delta = prev_loss - (epoch_loss/num_chunk)
        prev_loss = epoch_loss/num_chunk

        print('Current delta:', delta, 'Min delta:', min_delta)
        
        if delta <= min_delta:
            no_improv+=1
            print('No improvement for ', no_improv, ' epochs.')
            if no_improv < stop_epoch:
                epoch_loss += chunk_loss/(num_chunk+1)
                continue
            else:
                torch.save(best_l1, model_path+'dnn_l1.pth')
                print('Finished pretraining Layer 1...')
                break
        else:
            epoch_loss += chunk_loss/(num_chunk+1)
            torch.save(best_l1, model_path+'dnn_l1.pth')
            continue
        
    '''
    ###### TRAIN SECOND LAYER ##########

    l1 = Layer1()

    l1.load_state_dict(torch.load(model_path+'dnn_l1.pth'))
    
    #Remove the last layer
    newmodel = torch.nn.Sequential(*(list(l1.children())[:-2]))

    l2 = Layer_1_2(newmodel)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(l2.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda")
    l2.cuda()
    l2 = l2.to(device)
    criterion.cuda()

    best_l2 = copy.deepcopy(l2.state_dict())

    print("Start PRETRAINING second layer...")

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))

        epoch_loss = 0.0

        num_chunk = (4620//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 4620)
            print(start, end)

            X_chunk, y_chunk = make_batch(x_path, y_path, 
                                         [start, end], 5, 
                                         maxlen, win_len, 
                                         hop_size, feat_type, fs)

            trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 128)

            for step, (audio, target) in enumerate(trainData): 
                audio = audio.to(device)
                target = target.to(device)
                l2.train()
                output = l2(audio)
                newLoss = criterion(output,target)                
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()
            
            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(trainData)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        #Check for early stopping
        losses_l2.append(epoch_loss/num_chunk)
        pickle.dump(losses_l1, open(loss_path+"losses_l2.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))

        delta = prev_loss - (epoch_loss/num_chunk)
        prev_loss = epoch_loss/num_chunk

        print('Current delta:', delta, 'Min delta:', min_delta)
        
        if delta <= min_delta:
            no_improv+=1
            print('No improvement for ', no_improv, ' epochs.')
            if no_improv < stop_epoch:
                epoch_loss += chunk_loss/(num_chunk+1)
                continue
            else:
                torch.save(best_l2, model_path+'dnn_l2.pth')
                print('Finished pretraining Layer 2...')
                break
        else:
            epoch_loss += chunk_loss/(num_chunk+1)
            continue







def train_dnn(num_epochs, model_path, x_path, y_path, 
              loss_path, chunk_size, feat_type,
              maxlen=1339, win_len=512, hop_size=256, fs=16000, from_pretrained=False):
    if feat_type=='stft':
        model = DNN()
    elif feat_type=='mel':
        model = DNN_mel()
    model.apply(weights)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda")
    model.cuda()
    model = model.to(device)
    criterion.cuda()


    #Training loop

    best_model = copy.deepcopy(model.state_dict())
    losses = []

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        loss = 0.0 
        
        num_chunk = (4620//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 4620)
            print(start, end)

            X_chunk, y_chunk = make_batch(x_path, y_path, 
                                         [start, end], 5, 
                                         maxlen, win_len, 
                                         hop_size, feat_type, fs)

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

            chunk_loss = chunk_loss.detach().cpu().numpy()/len(trainData) 
            loss += chunk_loss           
            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))


        losses.append(loss/num_chunk)
        pickle.dump(losses, open(loss_path+"losses.p", "wb" ) )
        
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, loss/num_chunk))


        torch.save(best_model, model_path+'dnn_map_best.pth')



def inference(test_data_path,
              out_test, model_path, imag_path, 
              chunk_size, feat_type, maxlen=1339,
              win_len=512, hop_size=256, fs=16000):
    if feat_type=='stft':
        model = DNN()
    elif feat_type=='mel':
        model = DNN_mel()

    model.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))
    fnames = os.listdir(test_data_path)

    num_chunk = (1680//chunk_size) +1
    for chunk in range(num_chunk):
        chunk_loss = 0
        start = chunk*chunk_size
        end = min(start+chunk_size, 1680)
        print(start, end)
        x_list = [test_data_path + n for n in fnames]
        X_chunk = make_batch_test(x_list, [start, end], 5, feat_type, maxlen, win_len, hop_size, fs)
        print('X chunk:', X_chunk.shape)
        testData = data.DataLoader(testDataLoader(X_chunk), batch_size = 1339)
        print('len:', len(testData))

        chunk_names = fnames[start:end]
        #print('chunk names:', chunk_names)
        for step, audio in enumerate(testData):
            #print('Step:', step)

            name = chunk_names[step]
            #print('name:', name)
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
