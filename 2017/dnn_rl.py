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
            x_out (tensor): the output predicted by the extracted from q_func
                            Wiener filter
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

    def forward(self, x_out, x_source, x_clean):
        #Generate the ground truth labels
        A_t = []
        for timestep in range(x_source.shape[1]):
            sums = []
            for a in range(self.G_mat.shape[1]):
                diff = np.sum(x_clean[:,timestep] - np.multiply(self.G_mat[:,a], x_source[:, timestep]))
                sums.append(diff)
            sums = np.asarray(sums)
            A_t.append(np.argmin(sums))
        # Calculate the output with ground truth labels
        wiener_true = np.zeros((x_source.shape[0], x_source.shape[1]))
        for a in A_t:
            wiener_true[:,a] = self.G_mat[:, a]
        true_out = np.multiply(wiener_true, x_source).T
        true_out = torch.tensor(true_out).cuda().float()
        print(true_out.size())
        x_out = torch.tensor(x_out.T).cuda().float()
        print(x_out.size())
        loss =  nn.MSELoss()(true_out, x_out)
        print('Loss:', loss)



##### TRAINING FUNCTIONS #####

def q_training_step(output, step, G, criterion, x_path, clean_path, imag_path, fnames, maxlen=1339):
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
    phase = pad(np.load(imag_path+fnames[step]), maxlen).T
    print('Out size:', output.size())
    Q_pred = output.detach().cpu().numpy()
    wiener_rl = np.zeros((1339, 257))

    #Select template index, predict Wiener filter
    for i, row in enumerate(Q_pred):
        ind = np.argmax(row)
        G_k_pred = G[ind]
        wiener_rl[i] = G_k_pred

    x_source = np.abs(np.load(x_path+fnames[step]))
    x_source = pad(x_source, maxlen).T
  
    y_pred_rl = np.multiply(x_source, wiener_rl).T #+ phase
    #y_pred_rl = torch.tensor(y_pred_rl.T, requires_grad=True).cuda().float()

    clean = np.abs(pad(np.load(clean_path+fnames[step]), maxlen))
    #clean = torch.tensor(clean.T).cuda().float()
    #Loss input x_out, x_source, x_clean
    #x_source = torch.tensor(x_source.T).cuda().float()
    newLoss = criterion(y_pred_rl, x_source.T, clean)
    
    return newLoss


def MMSE_pretrain(chunk_size, x_path, y_path, model_path, cluster_path,
                clean_path,
                imag_path='/nobackup/anakuzne/data/snr0_train_img/',
                maxlen=1339, 
                win_len=512,
                hop_size=256, fs=16000):

    feat_type='mel'

    num_epochs = 100
    P=5 #Window size
    G = np.load(cluster_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 

    ##### EARLY STOPPING #####

    min_delta = 0.01 #Min change in loss which can be considered as improvement
    stop_epoch = 10 #Number of epochs without improvement
    no_improv = 0
    prev_loss = 0

    losses_l1 = []
    losses_l2 = []
   
    device = torch.device('cuda:0') #change to 2 if on Ada
    torch.cuda.set_device(0) #change to 2 if on Ada

    ######## PRETRAIN FIRST RL-LAYER #########

    l1 = RL_L1()
    l1.apply(weights)
    criterion = MMSE_loss(G)
    optimizer = optim.SGD(l1.parameters(), lr=0.001, momentum=0.9) #Changed lr for test
    
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
                                          x_path, clean_path, imag_path, fnames)               
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(trainData)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        #Check for early stopping
        losses_l1.append(epoch_loss/num_chunk)
        pickle.dump(losses_l1, open(model_path+"losses_l1.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))
        '''
        if epoch==1:
            prev_loss = epoch_loss/num_chunk
            epoch_loss += chunk_loss/(num_chunk+1)
            torch.save(best_l1, model_path+'dnn_rl_l1.pth')
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
                    torch.save(best_l1, model_path+'rl_dnn_l1.pth')
                    continue
                else:
                    prev_loss = 1
                    no_improv = 0
                    torch.save(best_l1, model_path+'rl_dnn_l1.pth')
                    print('Finished pretraining Layer 1...')
                    break
            else:
                epoch_loss += chunk_loss/(num_chunk+1)
                torch.save(best_l1, model_path+'qfunc_l1.pth')
                continue
    

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
                                          x_path, clean_path, imag_path, fnames)               
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

        if epoch==1:
            prev_loss = epoch_loss/num_chunk
            epoch_loss += chunk_loss/(num_chunk+1)
            torch.save(best_l2, model_path+'dnn_rl_l2.pth')
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
                    torch.save(best_l2, model_path+'rl_dnn_l2.pth')
                    continue
                else:
                    prev_loss = 1
                    no_improv = 0
                    torch.save(best_l2, model_path+'rl_dnn_l2.pth')
                    print('Finished pretraining Layer 2...')
                    break
            else:
                epoch_loss += chunk_loss/(num_chunk+1)
                torch.save(best_l2, model_path+'qfunc_l2.pth')
                continue
    '''

########################################################

def MMSE_train(chunk_size, x_path, y_path, model_path, cluster_path,
                clean_path,
                imag_path='/nobackup/anakuzne/data/snr0_train_img/',
                maxlen=1339, 
                win_len=512,
                hop_size=256, fs=16000):

    feat_type='mel'

    num_epochs = 100
    P=5 #Window size
    G = np.load(cluster_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 

    ##### EARLY STOPPING #####

    min_delta = 0.01 #Min change in loss which can be considered as improvement
    stop_epoch = 10 #Number of epochs without improvement
    no_improv = 0
    prev_loss = 0

    losses = []
    device = torch.device('cuda:2')
    torch.cuda.set_device(2)

    layers = RL_L2()
    layers.load_state_dict(model_path+'qfunc_l2.pth')

    q_func_MMSE = DNN_RL(layers)

    optimizer = optim.SGD(q_func_MMSE.parameters(), lr=0.01, momentum=0.9)
    q_func_MMSE.cuda()
    best_q = copy.deepcopy(q_func_MMSE.state_dict())


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
                output = q_func(audio)
                
                newLoss = q_training_step(output, step, G, criterion, 
                                          x_path, clean_path, imag_path, fnames)               
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(trainData)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        #Check for early stopping
        losses.append(epoch_loss/num_chunk)
        pickle.dump(losses, open(model_path+"losses_q.p", "wb" ) )
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))

        if epoch==1:
            prev_loss = epoch_loss/num_chunk
            epoch_loss += chunk_loss/(num_chunk+1)
            torch.save(best_l1, model_path+'q_func.pth')
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
                    torch.save(best_l2, model_path+'q_func.pth')
                    continue
                else:
                    prev_loss = 1
                    no_improv = 0
                    torch.save(best_l1, model_path+'q_func.pth')
                    break
            else:
                epoch_loss += chunk_loss/(num_chunk+1)
                torch.save(best_l1, model_path+'q_func.pth')
                continue

def q_learning(x_path, y_path, model_path, clean_path,
               imag_path='/nobackup/anakuzne/data/snr0_train_img/',
               num_episodes=50000, epsilon=0.01, maxlen=1339, 
               win_len=512,
               hop_size=256,
               fs=16000,
               from_pretrained=False):
    '''
    Params:
        x_path: path to the training examples
        y_path: path to the cluster centers
        model_path: path to dir where DNN-mapping model is stored
        clean_path: path to clean reference (stft)
    '''
    ### Initialization ###

    # Q-functions, zero initialization
    Q_target = np.zeros((1339, 32))
    Q_MMSE = np.zeros((1339, 32))

    P=5 #Window size
    G = np.load(y_path) #Cluster centers for wiener masks
    torch.cuda.empty_cache() 
    ###Load DNN-mapping model
    device = torch.device('cuda:2')
    torch.cuda.set_device(2)
    
    dnn_map = DNN_mel()
    dnn_map.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))
    dnn_map = dnn_map.to(device)
    
    
    dnn_rl = DNN_RL()
    dnn_rl.apply(weights)
    dnn_rl = dnn_rl.to(device)

    #criterion = nn.MSELoss()
    #optimizer = optim.SGD(dnn_rl.parameters(), lr=0.01, momentum=0.9)
    #device = torch.device('cuda:3')
    #device2 = torch.device('cuda:1')
    #dnn_rl.cuda()
    #dnn_rl = dnn_rl.to(device)
    #criterion.cuda()

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

    ### UPDATE TARGET Q-FUNCS ###
    R_ = R(z_rl, z_map)

    for i in range(r.shape[0]):
        if R_ > 0:
            Q_target[i][selected_actions[i]] = r[i] + max(Q_pred[i])
        else:
            Q_target[i][selected_actions[i]] = Q_pred[i][selected_actions[i]]


