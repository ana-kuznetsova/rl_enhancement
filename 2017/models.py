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
from dnn_rl import QDataSet

from data import make_windows
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



'''
class DNN_mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bnorm = nn.BatchNorm1d(704)
        self.fc1 = nn.Linear(704, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 257)
        self.drop = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.bnorm(x)
        x = Func.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = Func.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x
'''
class Layer1(nn.Module):
    '''
    Train with mel features
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(704, 128)
        #self.bnorm = nn.BatchNorm1d(704)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 257)

    def forward(self, x):
        #x = self.bnorm(x)
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        return self.out(x)

class Layer_1_2(nn.Module):
    def __init__(self, l1=None):
        super().__init__()
        if l1:
            self.fc1 = l1.fc1
        self.fc1 = nn.Linear(704, 128)
        self.fc2 = nn.Linear(128, 128)
        self.drop = nn.Dropout(0.3)
        #self.bnorm = nn.BatchNorm1d(704)
        self.out = nn.Linear(128, 257)

    def forward(self, x):
        #x = self.bnorm(x)
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        return self.out(x)

class DNN_mel(nn.Module):
    def __init__(self, l1_2=None):
        super().__init__()
        if l1_2:
            self.fc1 = l1_2.fc1
            self.fc2 = l1_2.fc2
        else:
            self.fc1 = nn.Linear(704, 128)
            self.fc2 = nn.Linear(128, 128)
        self.bnorm = nn.BatchNorm1d(704)
        self.fc3 = nn.Linear(128, 257)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.bnorm(x)
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x 
        

def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)


def pretrain(chunk_size, model_path, x_path, y_path, loss_path, num_epochs=50
             , win_len=512, hop_size=256, fs=16000):
    
    feat_type='stft'
    losses_l1 = []
    losses_l2 = []
    val_losses = []

    ############# PRETRAIN FIRST LAYER ################
    
    l1 = Layer1()
    l1.apply(weights)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(l1.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda")
    l1.cuda()
    l1 = l1.to(device)
    criterion.cuda()

    best_l1 = copy.deepcopy(l1.state_dict())

    print('---------------------------------')
    print("Start PRETRAINING first layer...")
    print('--------------------------------')

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))

        epoch_loss = 0.0

        num_chunk = (3234//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 3234)
            print(start, end)

            X_chunk, y_chunk, batch_indices = make_windows(x_path, y_path,
                                          [start, end], P, 
                                           win_len, 
                                           hop_size, fs, nn_type='map')

            #trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 128)
            dataset = QDataSet(X_chunk, y_chunk, batch_indices)
            loader = data.DataLoader(dataset, batch_size=1)

            for x, target in loader:
                x = x.to(device)
                x = x.reshape(x.shape[1], x.shape[2])
                target = target.to(device).long()
                target = torch.flatten(target)
                output = l1(x)

                newLoss = criterion(output, target)              
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(loader)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        losses_l1.append(epoch_loss/num_chunk)
        pickle.dump(losses_l1, open(loss_path+"losses_l1.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))

        #### VALIDATION #####
       
        print('Starting validation...')
        # Y is a clean speech spectrogram
        start = 3234
        end = 4620
        X_val, A_val, batch_indices = make_windows(x_path, y_path,
                                          [start, end], P, 
                                           win_len, 
                                           hop_size, fs)

        #valData = data.DataLoader(trainDataLoader(X_val, A_val), batch_size = 128)
        dataset = QDataSet(X_val, A_val, batch_indices)
        val_loader = data.DataLoader(dataset, batch_size=1)
        overall_val_loss=0

        for x, target in val_loader:
            x = x.to(device)
            x.requires_grad=True
            x = x.reshape(x.shape[1], x.shape[2])
            target = target.to(device).long()
            target = torch.flatten(target)
            output = l1(x)
            valLoss = criterion(x, target)
            overall_val_loss+=valLoss.detach().cpu().numpy()

        val_losses.append(overall_val_loss/len(val_loader))
        print('Validation loss: ', overall_val_loss/len(val_loader))
        np.save(model_path+'val_losses_l1.npy', np.asarray(val_losses))


        
        
    
    ###### TRAIN SECOND LAYER ##########

    l1 = Layer1()

    l1.load_state_dict(torch.load(model_path+'dnn_l1.pth'))

    l2 = Layer_1_2(l1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(l2.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda")
    l2.cuda()
    l2 = l2.to(device)
    criterion.cuda()

    best_l2 = copy.deepcopy(l2.state_dict())

    print('---------------------------------')
    print("Start PRETRAINING second layer...")
    print('---------------------------------')

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
        pickle.dump(losses_l2, open(loss_path+"losses_l2.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))

        if epoch==1:
            prev_loss = epoch_loss/num_chunk
            epoch_loss += chunk_loss/(num_chunk+1)
            torch.save(best_l1, model_path+'dnn_l2.pth')
            continue
        else:

            delta = prev_loss - (epoch_loss/num_chunk)
            prev_loss = epoch_loss/num_chunk

            print('Current delta:', delta, 'Min delta:', min_delta)
            
            if delta <= min_delta:
                no_improv+=1
                print('No improvement for ', no_improv, ' epochs.')
                if no_improv < stop_epoch:
                    epoch_loss += chunk_loss/(num_chunk+1)
                    torch.save(best_l2, model_path+'dnn_l2.pth')
                    continue
                else:
                    torch.save(best_l2, model_path+'dnn_l2.pth')
                    print('Finished pretraining Layer 2...')
                    break
            else:
                epoch_loss += chunk_loss/(num_chunk+1)
                torch.save(best_l2, model_path+'dnn_l2.pth')
                continue


def train_dnn(num_epochs, model_path, x_path, y_path, 
              loss_path, chunk_size, feat_type, pretrain_path, from_pretrained=False,
              maxlen=1339, win_len=512, hop_size=256, fs=16000):
    if feat_type=='stft':
        model = DNN()
    elif feat_type=='mel':
        if from_pretrained:
            l1 = Layer1()
            l1.load_state_dict(torch.load(pretrain_path+'dnn_l1.pth'))
            l1_2 = Layer_1_2(l1)
            l1_2.load_state_dict(torch.load(pretrain_path+'dnn_l2.pth'))
            model = DNN_mel(l1_2)
        
        else:
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
              chunk_size, feat_type, mask='ln', maxlen=1339,
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
        testData = data.DataLoader(testDataLoader(X_chunk), batch_size = 1339)

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
                if mask=='ln':
                    np.save(out_test+name, np.exp(output)+imag)
                elif mask=='wiener':
                    noisy_aud = pad(np.load(test_data_path+name), maxlen)
                    result = np.multiply(output, noisy_aud)
                    np.save(out_test+name, result+imag)
