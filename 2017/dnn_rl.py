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
from data import make_windows
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
    def __init__(self, l1_2=None):
        super().__init__()
        if l1_2:
            self.fc1 = l1_2.fc1
            self.fc2 = l1_2.fc2
        else:
            self.fc1 = nn.Linear(704, 64)
            self.fc2 = nn.Linear(64, 32)
        
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
            a_target (tensor): true action labels
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x_out, a_target):
        loss = nn.CrossEntropyLoss()
        new_loss = loss(x_out, a_target)
        return new_loss



##### TRAINING FUNCTIONS #####

def q_training_step(output, step, G, criterion, x_path, a_path, clean_path, imag_path, 
                    fnames, maxlen=1339, proc='train'):
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
    action_labels = np.load(a_path+fnames[step])
    new_loss = criterion(output,)
    if proc=='train':
        return new_loss
    elif proc=='val':
        return new_loss, action_labels


def MMSE_pretrain(chunk_size, x_path, a_path, model_path, cluster_path,
                clean_path,
                imag_path='/nobackup/anakuzne/data/snr0_train_img/',
                maxlen=1339, 
                win_len=512,
                hop_size=256, fs=16000):

    num_epochs = 50
    P=5 #Window size
    #G = np.load(cluster_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 

    losses_l1 = []
    val_losses = []
    true_actions = []
    pred_actions = []
    #losses_l2 = []
   
    device = torch.device('cuda:0') #change to 2 if on Ada
    torch.cuda.set_device(0) #change to 2 if on Ada

    ######## PRETRAIN FIRST RL-LAYER #########

    l1 = RL_L1()
    l1.apply(weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(l1.parameters(), lr=0.001, momentum=0.8) #Changed lr for test
    
    l1.cuda()
    l1 = l1.to(device)
    criterion.cuda()

    best_l1 = copy.deepcopy(l1.state_dict())

    print('###### Pretraining RL_L1 #######')

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0.0
        labels = []
        ##Training 
        num_chunk = (3234//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 3234)
            print(start, end)
            #returns both training examples and true labels 
            X_chunk, A_chunk = make_windows(x_path, a_path,
                                          [start, end], P, 
                                           win_len, 
                                           hop_size, fs)
            
            labels.extend(A_chunk)
            trainData = data.DataLoader(trainDataLoader(X_chunk, A_chunk), batch_size = 128)

            for x, target in trainData:
                x = x.to(device)
                target = target.to(device).long()
                target = torch.flatten(target)
                output = l1(x)
                pred_actions.append(output.detach().cpu().numpy())
                newLoss = criterion(output, target)              
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()


            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(X_chunk)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        
        losses_l1.append(epoch_loss/num_chunk)
        pickle.dump(losses_l1, open(model_path+"losses_l1.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))

        ##Validation
        print('Starting validation...')
        # Y is a clean speech spectrogram
        start = 3234
        end = 4620
        X_val, A_val = make_windows(x_path, a_path,
                                          [start, end], P, 
                                           win_len, 
                                           hop_size, fs)

        valData = data.DataLoader(trainDataLoader(X_val, A_val), batch_size = 128)
        overall_val_loss=0

        for x, target in trainData:
            x = x.to(device)
            x.requires_grad=True
            target = target.to(device).long()
            target = torch.flatten(target)
            output = l1(x)
            valLoss = criterion(x, target)
            overall_val_loss+=valLoss.detach().cpu().numpy()

        val_losses.append(overall_val_loss/len(valData))
        print('Validation loss: ', overall_val_loss/len(valData))
        np.save(model_path+'val_losses_l1.npy', np.asarray(val_losses))
        true_actions.append(labels)
        np.save(model_path+'true_actions_l1.npy', np.asarray(true_actions))
        np.save(model_path+'pred_actions_l1.npy', np.asarray(pred_actions))


        print('Saing model...')
        torch.save(best_l1, model_path+'rl_dnn_l1.pth')

    '''
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
   '''

########################################################

def MMSE_train(chunk_size, x_path, y_path, a_path, model_path, cluster_path,
                clean_path,
                imag_path='/nobackup/anakuzne/data/snr0_train_img/',
                maxlen=1339, 
                win_len=512,
                hop_size=256, fs=16000):

    feat_type='mel'

    num_epochs = 5
    P=5 #Window size
    G = np.load(cluster_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 

    losses = []
    val_losses = []
    pred_actions = []
    true_actions = []

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

        ##Training 
        num_chunk = (3234//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 3234)
            print(start, end)

            # Y is a clean speech spectrogram
            X_chunk, y_chunk, fnames = make_batch(x_path, y_path, 
                                         [start, end], P, 
                                         maxlen, win_len, 
                                         hop_size, feat_type, fs, names=True)
            
            trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 1339)

            for step, (audio, target) in enumerate(trainData): 
                audio = audio.to(device)
                target = target.to(device)
                output = q_func_pretrained(audio)
                
                newLoss = q_training_step(output, step, G, criterion, 
                                          x_path, a_path, clean_path, imag_path, fnames, proc='train')               
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

        ##Validation
        print('Starting validation...')
        # Y is a clean speech spectrogram
        start = 3234
        end = 4620
        X_val, y_val, fnames = make_batch(x_path, y_path, 
                                         [start, end], 5, 
                                         maxlen, win_len, 
                                         hop_size, feat_type, fs, names=True)
            
        valData = data.DataLoader(trainDataLoader(X_val, y_val), batch_size = 1339)

        overall_val_loss=0
        for step, (audio, target) in enumerate(valData): 
            audio = audio.to(device)
            target = target.to(device)
            output = q_func_pretrained(audio)
            
            valLoss, labels = q_training_step(output, step, G, criterion, 
                                     x_path, a_path, clean_path, imag_path, fnames, proc='val') 
            overall_val_loss+=valLoss.detach().cpu().numpy()
        val_losses.append(overall_val_loss/len(valData))
        print('Validation loss: ', overall_val_loss/len(valData))
        np.save(model_path+'val_losses.npy', np.asarray(val_losses))
        true_actions.append(labels)
        np.save(model_path+'true_actions.npy', np.asarray(true_actions))
        pred_actions.append(output.detach().cpu().numpy())
        np.save(model_path+'pred_actions.npy', np.asarray(pred_actions))
        

    print('Saving model...')
    torch.save(best_q, model_path+'qfunc_pretrained.pth')


def q_learning(num_episodes, x_path, cluster_path, model_path, clean_path,
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
    G = np.load(cluster_path) #Cluster centers for wiener masks
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

    ##Loss
    criterion = nn.MSELoss()
    opt_RMSprop = optim.RMSprop(dnn_rl.parameters(), lr = 0.001, alpha = 0.9)
    #optimizer = optim.SGD(l1.parameters(), lr=0.01, momentum=0.9)
    criterion.cuda()

    ## Initialize qfunc matrices
    qfunc_target = np.zeros((1339, 32)) 
    qfunc_pretrained = np.zeros((1339, 32))

    q_losses = []
    reward_sums = []
    
    for ep in range(num_episodes):
        if ep//100:
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

        Q_pred_rl = dnn_rl(x).detach().cpu().numpy() #for target Qfunc. Shape (1339, 32)
        Q_pred_mmse = q_func_mmse(x).detach().cpu().numpy() #for pretrained Qfunc
        wiener_rl = np.zeros((1339, 257))

        #Save selected actions
        selected_actions_target = []
        selected_actions_mmse = []
    
        #Select template index, predict Wiener filter
        for i, row in enumerate(Q_pred_rl):
        #E-greedy selection for target
            a = np.array([0,1])
            probs = np.array([epsilon, 1-epsilon])
            strategy = np.random.choice(a, p=probs)
            if strategy==0:
                ind_t = np.random.choice(np.arange(32))
            else:
                ind_t = np.argmax(row)
            ind_m = np.argmax(Q_pred_mmse[i])
            selected_actions_target.append(ind_t)
            selected_actions_mmse.append(ind_m)
            G_k_pred = G[ind_t]
            wiener_rl[i] = G_k_pred

        wiener_rl = wiener_rl.T
        y_pred_rl = np.multiply(pad(x_source, maxlen), wiener_rl) + phase  

        map_out = dnn_map(x)
        wiener_map = map_out.detach().cpu().numpy().T
        y_pred_map = np.multiply(pad(x_source, maxlen), wiener_map) + phase  

    
        ##### Calculate reward ######

        x_source_wav = invert(x_source)
        y_map_wav = invert(y_pred_map)[:x_source_wav.shape[0]]
        y_rl_wav = invert(y_pred_rl)[:x_source_wav.shape[0]]
        
        z_rl = calc_Z(x_source_wav, y_rl_wav)
        z_map = calc_Z(x_source_wav, y_map_wav)
        #print('Z-scores:', z_rl, z_map)

        clean = np.load(clean_path+x_name)
        E = time_weight(y_pred_rl, pad(clean, maxlen))
        r = reward(z_rl, z_map, E)
        #If inf in reward, skip iter
        if np.isnan(np.sum(r)):
            continue
        reward_sums.append(np.sum(r))
        np.save(model_path+'reward_sum.npy', np.asarray(reward_sums))
        print('Reward sum:', np.sum(r))
        
        R_ = R(z_rl, z_map)
        #print('R_cal:', R_)

        #### UPDATE Q-FUNCS ####

        for i, a_t in enumerate(selected_actions_target):
            a_m = selected_actions_mmse[i]
            if a_t==a_m:
                if R_ > 0:
                    qfunc_target[i][a_t] = r[i] + np.max(qfunc_target[i]) #qfunc shape (1339, 32)
                else:
                    qfunc_target[i][a_t] = Q_pred_rl[i][a_t]
            else:
                if R_ > 0:
                    qfunc_pretrained[i][a_m] = Q_pred_mmse[i][a_m]
                else:
                    qfunc_pretrained[i][a_m] = Q_pred_mmse[i][a_m] - r[i]

        target_tensor = torch.tensor(qfunc_target, requires_grad=True).cuda().float()
        pretrained_tensor = torch.tensor(qfunc_pretrained).cuda().float()
        dnn_rl.train()
        curr_loss = criterion(target_tensor, pretrained_tensor)
        q_losses.append(curr_loss.detach().cpu().numpy())
        np.save(model_path+'q_losses.npy', q_losses)

        print('Episode {}, Training loss:{:>4f}'.format(ep, curr_loss.detach().cpu().numpy()))
        opt_RMSprop.zero_grad()
        curr_loss.backward()
        opt_RMSprop.step()