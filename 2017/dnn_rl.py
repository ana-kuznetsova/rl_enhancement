import os
import numpy as np
import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from models import weights
from models import DNN_mel
from models import trainDataLoader

from data import mel_spec
from data import pad
from data import get_X_batch
from data import make_batch
from utils import invert
from metrics import calc_Z


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
        self.fc1 = nn.Linear(704, 64)
        self.fc2 = nn.Linear(64, 32)
        self.drop = nn.Dropout(0.3)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        return self.soft(x)
    
class DNN_RL(nn.Module):
    def __init__(self, l1_2=None, inference=False):
        super().__init__()
        if l1_2:
            self.fc1 = l1_2.fc1
            self.fc2 = l1_2.fc2
        else:
            self.fc1 = nn.Linear(704, 64)
            self.fc2 = nn.Linear(64, 32)
        if inference:
            self.soft = l1_2.soft
        else:
            self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.soft(x)
        return x 

#### REWARD DEFINITION ####

def reward(z_rl, z_map, E):
    '''
    z_rl: predicted G from DNN-RL
    z_map: predicted G from DNN-map
    '''
    R_ = R(z_rl, z_map)
    if R_ > 0:
        return (1 - E)*R_
    else:
        return E*R_


def R(z_rl, z_map, alpha=20):
    return np.tanh(alpha*(z_rl - z_map))

def time_weight(Y, S):
    '''
    E - calculated time weight
    Y: predicted spectrogram
    S: true spectrogram
    '''
    Y = np.nan_to_num(np.log(np.abs(Y)))
    S = np.nan_to_num(np.log(np.abs(S)))
    sum_ = np.nan_to_num((Y - S)**2)
    E_approx = np.nan_to_num(np.sum(sum_, axis=0))
    E = E_approx/np.max(E_approx)
    return E

##### MMSE loss ####
class MMSE_loss(torch.nn.Module):
    '''
        Params:
            x_out (tensor): predicted q-function
            x_source (tensor): noisy mixture of the signal
            x_clean (tensor): clean speech
            G_mat: cluster centers
        We are trying to fit the template s.t. minimizes error in q-func
        by calculating the ground truth labels of actions
    '''
    def __init__(self, G_mat):
        super().__init__()
        #self.G_mat = torch.tensor(G_mat.T).cuda().float()
        self.G_mat = G_mat.T

    def forward(self, x_out, x_source, x_clean, action_labels):
        q_target = torch.tensor(action_labels).cuda()
        loss = nn.CrossEntropyLoss()
        new_loss = loss(x_out, q_target)
        return new_loss



##### TRAINING FUNCTIONS #####

def q_training_step(output, step, G, criterion, x_path, a_path, clean_path, imag_path, 
                    fnames, maxlen=1339):
    '''
    Params:
        output: NN predictions
        step: step index
        G: cluster centers
        criterion
        x_path
        clean_path
        imag_path
        fnames
    '''
    x_source = np.abs(np.load(x_path+fnames[step]))
    x_source = pad(x_source, maxlen).T
    
    clean = np.abs(pad(np.load(clean_path+fnames[step]), maxlen))
    action_labels = np.load(a_path+fnames[step])
    new_loss = criterion(output, x_source.T, clean, action_labels)
    return new_loss


def MMSE_pretrain(chunk_size, x_path, y_path, a_path, model_path, cluster_path,
                clean_path,
                imag_path='/nobackup/anakuzne/data/snr0_train_img/',
                maxlen=1339, 
                win_len=512,
                hop_size=256, fs=16000):

    feat_type='mel'

    num_epochs = 10
    P=5 #Window size
    G = np.load(cluster_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 

    losses_l1 = []
    losses_l2 = []
   
    device = torch.device('cuda:0') #change to 2 if on Ada
    torch.cuda.set_device(0) #change to 2 if on Ada

    ######## PRETRAIN FIRST RL-LAYER #########

    l1 = RL_L1()
    l1.apply(weights)
    criterion = MMSE_loss(G)
    optimizer = optim.SGD(l1.parameters(), lr=0.0001, momentum=0.8) #Changed lr for test
    
    l1.cuda()
    l1 = l1.to(device)
    criterion.cuda()

    best_l1 = copy.deepcopy(l1.state_dict())

    print('###### Pretraining RL_L1 #######')

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0.0

        num_chunk = (4620//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 4620)
            print(start, end)

            # Y is a clean speech spectrogram
            X_chunk, y_chunk, fnames = make_batch(x_path, y_path, 
                                         [start, end], 5, 
                                         maxlen, win_len, 
                                         hop_size, feat_type, fs, names=True)
            
            trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 1339)

            for step, (audio, target) in enumerate(trainData): 
                audio = audio.to(device)
                target = target.to(device)
                output = l1(audio)
                
                newLoss = q_training_step(output, step, G, criterion, 
                                          x_path, a_path, clean_path, imag_path, fnames)               
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(trainData)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        
        losses_l1.append(epoch_loss/num_chunk)
        pickle.dump(losses_l1, open(model_path+"losses_l1.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))
    print('Saved pre-trained L1...')
    torch.save(best_l1, model_path+'rl_dnn_l1.pth')

    ######## PRETRAIN SECOND LAYER ############

    l1 = RL_L1()

    l1.load_state_dict(torch.load(model_path+'rl_dnn_l1.pth'))

    l2 = RL_L2()
    optimizer = optim.SGD(l2.parameters(), lr=0.01, momentum=0.9)
    l2.cuda()
    best_l2 = copy.deepcopy(l2.state_dict())

    print('###### Pretraining RL_L2 #######')

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0.0

        num_chunk = (4620//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 4620)
            print(start, end)

            # Y is a clean speech spectrogram
            X_chunk, y_chunk, fnames = make_batch(x_path, y_path, 
                                         [start, end], 5, 
                                         maxlen, win_len, 
                                         hop_size, feat_type, fs, names=True)
            
            trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 1339)

            for step, (audio, target) in enumerate(trainData): 
                audio = audio.to(device)
                target = target.to(device)
                output = l2(audio)
                
                newLoss = q_training_step(output, step, G, criterion, 
                                          x_path, a_path, clean_path, imag_path, fnames)               
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(trainData)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        #Check for early stopping
        losses_l2.append(epoch_loss/num_chunk)
        pickle.dump(losses_l2, open(model_path+"losses_l2.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))
    print('Saved best L2...')
    torch.save(best_l2, model_path+'dnn_rl_l2.pth')
   

########################################################

def MMSE_train(chunk_size, x_path, y_path, a_path, model_path, cluster_path,
                clean_path,
                imag_path='/nobackup/anakuzne/data/snr0_train_img/',
                maxlen=1339, 
                win_len=512,
                hop_size=256, fs=16000):

    feat_type='mel'

    num_epochs = 50
    P=5 #Window size
    G = np.load(cluster_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 

    losses = []
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)

    layers = RL_L2()
    layers.load_state_dict(torch.load(model_path+'dnn_rl_l2.pth'))

    q_func_pretrained = DNN_RL(layers)

    criterion = MMSE_loss(G)
    criterion.cuda()

    optimizer = optim.SGD(q_func_pretrained.parameters(), lr=0.0001, momentum=0.8)
    q_func_pretrained.cuda()
    best_q = copy.deepcopy(q_func_pretrained.state_dict())

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0.0

        num_chunk = (4620//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 4620)
            print(start, end)

            # Y is a clean speech spectrogram
            X_chunk, y_chunk, fnames = make_batch(x_path, y_path, 
                                         [start, end], 5, 
                                         maxlen, win_len, 
                                         hop_size, feat_type, fs, names=True)
            
            trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 1339)

            for step, (audio, target) in enumerate(trainData): 
                audio = audio.to(device)
                target = target.to(device)
                output = q_func_pretrained(audio)
                
                newLoss = q_training_step(output, step, G, criterion, 
                                          x_path, a_path, clean_path, imag_path, fnames)               
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(trainData)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

       
        losses.append(epoch_loss/num_chunk)
        pickle.dump(losses, open(model_path+"losses_pre.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))
    print('Saved best model...')
    torch.save(best_q, model_path+'qfunc_pretrained.pth')


def q_learning(num_episodes, x_path, y_path, a_path, model_path, clean_path,
               imag_path='/nobackup/anakuzne/data/snr0_train_img/',
               epsilon=0.01, maxlen=1339, 
               win_len=512,
               hop_size=256,
               fs=16000,
               from_pretrained=False):
    '''
    Params:
        num_episodes: num of updates to the target q function
        x_path: path to the training examples
        y_path: path to the cluster centers
        a_path: path where ground truth actions are stored
        model_path: path where the models are stored
        clean_path: path to clean reference (stft)
    '''
    ### Initialization ###

    P=5 #Window size
    G = np.load(y_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 

    ###Load DNN-mapping model
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    
    dnn_map = DNN_mel()
    dnn_map.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))
    dnn_map.cuda()
    dnn_map = dnn_map.to(device)

    #Load MMSE reference Q-function
    q_func_mmse = DNN_RL()
    q_func_mmse.load_state_dict(torch.load(model_path+'qfunc_pretrained.pth'))
    q_func_mmse.cuda()
    q_func_mmse.to(device)
    
    #Initialize DNN_RL with pretrained weights
    
    dnn_rl = DNN_RL()
    dnn_rl.load_state_dict(torch.load(model_path+'qfunc_pretrained.pth'))
    dnn_rl.cuda()
    dnn_rl = dnn_rl.to(device)

    for ep in range(num_episodes):

        print('Episode:{}/{}'.format(ep, num_episodes))

        #Select random
        x_files = os.listdir(x_path)
        x_name = np.random.choice(x_files)

        phase = pad(np.load(imag_path+x_name), maxlen)

        x_source = np.load(x_path+x_name)
        x = mel_spec(x_source, win_len, hop_size, fs)
        x = np.abs(get_X_batch(x, P)).T
        x = pad(x, maxlen).T
        x = torch.tensor(x).cuda().float()

    ####### PREDICT DNN-RL AND DNN-MAPPING OUTPUT #######

        Q_pred = dnn_rl(x).detach().cpu().numpy() #Q_pred - q-function predicted by DNN-RL [1339, 32]
        wiener_rl = np.zeros((1339, 257))

        #Save selected actions
        selected_actions = []
    
        #Select template index, predict Wiener filter
        for i, row in enumerate(Q_pred):
            ind = np.argmax(row)
            selected_actions.append(ind)
            G_k_pred = G[ind]
            wiener_rl[i] = G_k_pred

        wiener_rl = wiener_rl.T
        y_pred_rl = np.multiply(pad(x_source, maxlen), wiener_rl) + phase  
        print('Pred shape:', y_pred_rl.shape)

        map_out = dnn_map(x)
        wiener_map = map_out.detach().cpu().numpy().T
        y_pred_map = np.multiply(pad(x_source, maxlen), wiener_map) + phase  

    
        ##### Calculate reward ######

        x_source_wav = invert(x_source)
        y_map_wav = invert(y_pred_map)[:x_source_wav.shape[0]]
        y_rl_wav = invert(y_pred_map)[:x_source_wav.shape[0]]
        
        z_rl = calc_Z(x_source_wav, y_rl_wav)
        z_map = calc_Z(x_source_wav, y_map_wav)
        print('Z-scores:', z_rl, z_map)

        clean = np.load(clean_path+x_name)
        E = time_weight(y_pred_rl, pad(clean, maxlen))
        r = reward(z_rl, z_map, E)
        print('Reward:', r)
        
        R_ = R(z_rl, z_map)

        #### UPDATE Q-FUNCS ####