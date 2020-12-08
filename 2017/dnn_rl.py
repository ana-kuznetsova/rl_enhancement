import os
import numpy as np
import copy
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from models import weights
from preproc import q_transform


class QDnnLoader(data.Dataset):
    def __init__(self, x_path, noise_path, cluster_path, snr, P, transform, mode='Train'):
        '''
        Args:
            x_path: path to the location where all the wav files stored
            noise_path: path to noise signal
            snr: desired snr
            P: window length
            transforms: list of transforms done with input
            mode: Train or Val. If train, take 0.7 of the data set
                  If validation take other 0.3.
        '''

        self.x_path = x_path
        self.noise_path = noise_path
        self.cluster_path = cluster_path
        self.snr = snr
        self.P = P
        self.mode = mode
        self.fnames = os.listdir(x_path)
        self.transform = transform
        self.train_fnames = self.fnames[:int(len(self.fnames)*0.7)]
        self.val_fnames = self.fnames[int(len(self.fnames)*0.7):]

    def __len__(self):
        if self.mode=='Train':
            return int(len(self.fnames)*0.7)
        else:
            return len(self.fnames) - int(len(self.fnames)*0.7)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode=='Train':
            fpath = os.path.join(self.x_path, self.train_fnames[idx])
        elif self.mode=='Val':
            fpath = os.path.join(self.x_path, self.val_fnames[idx])
        sample = self.transform(fpath, self.noise_path, self.cluster_path, self.snr, self.P)
        return sample


class QTestLoader(data.Dataset):
    def __init__(self, x_path, noise_path, cluster_path, snr, P, transform):
        '''
        Args:
            x_path: path to the location where all the wav files stored
            noise_path: path to noise signal
            snr: desired snr
            P: window length
            transforms: list of transforms done with input
        '''

        self.x_path = x_path
        self.noise_path = noise_path
        self.cluster_path = cluster_path
        self.snr = snr
        self.P = P
        self.fnames = os.listdir(x_path)
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fpath = os.path.join(self.x_path, self.fnames[idx])
        sample = self.transform(fpath, self.noise_path, self.cluster_path, self.snr, self.P)
        return sample

#### LAYERS FOR RL PRETRAINING ###

class RL_L1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(704, 64)
        self.drop = nn.Dropout(0.3)
        self.soft = nn.Softmax(dim=1)
        self.out = nn.Linear(64, 32)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = self.out(x)
        return self.soft(x)


class RL_L2(nn.Module):
    def __init__(self, l1=None):
        super().__init__()
        if l1:
            self.fc1 = l1.fc1
        else:
            self.fc1 = nn.Linear(704, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 32)
        self.drop = nn.Dropout(0.3)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.out(x)
        return self.soft(x)
    
class DNN_RL(nn.Module):
    def __init__(self, l1_2=None):
        super().__init__()
        if l1_2:
            self.fc1 = l1_2.fc1
            self.fc2 = l1_2.fc2
        else:
            self.fc1 = nn.Linear(704, 64)
            self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 32)
        self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.out(x)
        x = self.soft(x)
        return x 

##### TRAINING FUNCTIONS #####

def q_pretrain(x_path, noise_path, cluster_path, model_path, 
               num_epochs=100, snr=0, P=5, maxlen=1339, resume='False'):
    P=5 #Window size
    num_epochs = 50
    snr=0
    torch.cuda.empty_cache() 

    losses_l1 = []
    val_losses = []
    losses_l2 = []

    prev_val = 99999
   
    device = torch.device('cuda:0') #change to 2 if on Ada
    torch.cuda.set_device(0) #change to 2 if on Ada
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    if resume=='False':
    ######## PRETRAIN FIRST RL-LAYER #########

        l1 = RL_L1()
        l1.apply(weights)
    
        l1.cuda()
        l1 = l1.to(device).double()
        criterion.cuda()

        optimizer = optim.SGD(l1.parameters(), lr=0.001, momentum=0.9)

        best_l1 = copy.deepcopy(l1.state_dict())

        print('###### Pretraining RL_L1 #######')
      
        for epoch in range(1, num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            epoch_loss = 0.0

            dataset = QDnnLoader(x_path, noise_path, cluster_path, snr, P, q_transform, 'Train')
            loader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
            
            for batch in loader:
                x = batch['x']
                x = x.to(device)
                target = batch['t']
                target = target.to(device).long()
                output = l1(x)
                output = torch.transpose(output, 1, 2)   
                newLoss = criterion(output, target.squeeze(2))    
                epoch_loss += newLoss.detach().cpu().numpy()
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()
            
            losses_l1.append(epoch_loss/len(loader))
            np.save(os.path.join(model_path, "qlosses_l1.npy"), np.array(losses_l1))
            
            print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))

            ##Validation
            print('Starting validation...') 
            
            dataset = QDnnLoader(x_path, noise_path, cluster_path, snr, P, q_transform, 'Val')
            val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
            overall_val_loss=0

            for batch in val_loader:
                x = batch['x']
                x = x.to(device)
                target = batch['t']
                target = target.to(device).long()
                output = l1(x)
                output = torch.transpose(output, 1, 2)
                valLoss = criterion(output, target.squeeze(2))
                overall_val_loss+=valLoss.detach().cpu().numpy()


            curr_val_loss = overall_val_loss/len(val_loader)
            val_losses.append(curr_val_loss)
            print('Validation loss: ', curr_val_loss)
            np.save(model_path+'val_losses_l1.npy', np.array(val_losses))

            if curr_val_loss < prev_val:
                torch.save(best_l1, model_path+'rl_dnn_l1_best.pth')
                prev_val = curr_val_loss
            
            ##Save last model
            torch.save(best_l1, model_path+'rl_dnn_l1_last.pth')

        prev_val = 999999
        val_losses = []

    ######## PRETRAIN SECOND LAYER ############
    
    l1 = RL_L1()
    l1.load_state_dict(torch.load(model_path+'rl_dnn_l1_best.pth'))
    l2 = RL_L2().double()
    optimizer = optim.SGD(l2.parameters(), lr=0.01, momentum=0.9)
    l2.cuda()
    best_l2 = copy.deepcopy(l2.state_dict())

    print('###### Pretraining RL_L2 #######')

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0.0

        dataset = QDnnLoader(x_path, noise_path, cluster_path, snr, P, q_transform, 'Train')
        loader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        for batch in loader:
            x = batch['x']
            x = x.to(device)
            target = batch['t']
            target = target.to(device).long()
            output = l2(x)
            output = torch.transpose(output, 1, 2)   
            newLoss = criterion(output, target.squeeze(2))    
            epoch_loss += newLoss.data.detach().cpu().numpy()
            optimizer.zero_grad()
            newLoss.backward()
            optimizer.step()

        losses_l2.append(epoch_loss/len(loader))
        np.save(os.path.join(model_path, "qlosses_l2.npy"), np.array(losses_l2))
            
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))

        ##Validation
        print('Starting validation...')

        dataset = QDnnLoader(x_path, noise_path, cluster_path, snr, P, q_transform, 'Val')
        val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
        overall_val_loss=0

        for batch in val_loader:
            x = batch['x']
            x = x.to(device)
            target = batch['t']
            target = target.to(device).long()
            output = l2(x)
            output = torch.transpose(output, 1, 2)
            valLoss = criterion(output, target.squeeze(2))
            overall_val_loss+=valLoss.detach().cpu().numpy()
        
        curr_val_loss = overall_val_loss/len(val_loader)
        val_losses.append(curr_val_loss)
        print('Validation loss: ', curr_val_loss)
        np.save(model_path+'val_losses_l2.npy', np.array(val_losses))

        if curr_val_loss < prev_val:
            torch.save(best_l2, model_path+'rl_dnn_l2_best.pth')
            prev_val = curr_val_loss
        
        ##Save last model
        torch.save(best_l2, model_path+'rl_dnn_l2_last.pth')
    
########################################################

def q_train(x_path, noise_path, cluster_path, model_path, 
            num_epochs=100, snr=0, P=5, maxlen=1339, resume=False):

    torch.cuda.empty_cache() 
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    losses = []
    val_losses = []
    prev_val = 99999

    layers = RL_L2()
    layers.load_state_dict(torch.load(model_path+'rl_dnn_l2_best.pth'))

    q_func_pretrained = DNN_RL(layers)

    optimizer = optim.SGD(q_func_pretrained.parameters(), lr=0.0001, momentum=0.8)
    q_func_pretrained.cuda()
    q_func_pretrained = q_func_pretrained.double()
    best_q = copy.deepcopy(q_func_pretrained.state_dict())

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0.0


        dataset = QDnnLoader(x_path, noise_path, cluster_path, snr, P, q_transform, 'Train')
        loader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        for batch in loader:
            x = batch['x']
            x = x.to(device)
            target = batch['t']
            target = target.to(device).long()
            output = q_func_pretrained(x)
            output = torch.transpose(output, 1, 2)   
            newLoss = criterion(output, target.squeeze(2))    
            epoch_loss += newLoss.data.detach().cpu().numpy()
            optimizer.zero_grad()
            newLoss.backward()
            optimizer.step()

        losses.append(epoch_loss/len(loader))
        np.save(os.path.join(model_path, "q_pretrain_losses.npy"), np.array(losses))
            
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))

        ##Validation
        print('Starting validation...')

        dataset = QDnnLoader(x_path, noise_path, cluster_path, snr, P, q_transform, 'Val')
        val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
        overall_val_loss=0

        for batch in val_loader:
            x = batch['x']
            x = x.to(device)
            target = batch['t']
            target = target.to(device).long()
            output = q_func_pretrained(x)
            output = torch.transpose(output, 1, 2)
            valLoss = criterion(output, target.squeeze(2))
            overall_val_loss+=valLoss.detach().cpu().numpy()
        
        curr_val_loss = overall_val_loss/len(val_loader)
        val_losses.append(curr_val_loss)
        print('Validation loss: ', curr_val_loss)
        np.save(model_path+'qpretrain_val_losses.npy', np.asarray(val_losses))

        if curr_val_loss < prev_val:
            torch.save(best_q, model_path+'rl_dnn_best.pth')
            prev_val = curr_val_loss
        
        ##Save last model
        torch.save(best_q, model_path+'rl_dnn_last.pth')



def eval_actions(x_path, noise_path, cluster_path, model_path, snr=0, P=5):
    torch.cuda.empty_cache() 
    device = torch.device('cuda') 

    q_func_pretrained = DNN_RL()
    q_func_pretrained.load_state_dict(torch.load(model_path+'rl_dnn_best.pth'))
    q_func_pretrained.cuda()

    dataset = QTestLoader(x_path, noise_path, cluster_path, snr, P, q_transform)
    val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    pred_actions = []
    true_actions = []

    for batch in val_loader:
        x = batch['x']
        x = x.to(device)
        target = batch['t']
        target = target.to(device).long()
        output = q_func_pretrained(x)
        target = torch.flatten(target).detach().cpu().numpy()
        pred_qfunc = output.detach().cpu().numpy()

    
        for i in range(pred_qfunc.shape[0]):
            pred_actions.append(int(np.argmax(pred_qfunc[i]))) 
    
        for a in target:
            true_actions.append(int(a))
    
    np.save(model_path+"true_actions.npy", np.asarray(true_actions))
    np.save(model_path+"pred_actions.npy", np.asarray(pred_actions))