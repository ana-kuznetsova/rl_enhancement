import numpy as np
import copy
import os
import tqdm as tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


from preproc import make_dnn_feats
from preproc import invert_mel


########DATA LOADERS ########

class DnnLoader(data.Dataset):
    def __init__(self, x_path, noise_path, snr, P, transform, mode='Train'):
        '''
        Args:
            x_path: path to the location where all the wav files stored
            noise_path: path to noise signal
            snr: desired snr
            P: window length
            transform: func for feature generation
            mode: Train or Val. If train, take 0.7 of the data set
                  If validation take other 0.3.
        '''

        self.x_path = x_path
        self.noise_path = noise_path
        self.transform = transform
        self.snr = snr
        self.P = P
        self.mode = mode
        self.fnames = os.listdir(x_path)
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
        sample = self.transform(fpath, self.noise_path, self.snr, self.P)
        return sample

class DnnTestLoader(data.Dataset):
    def __init__(self, x_path, noise_path, snr, P, transform):
        '''
        Args:
            x_path: path to the location where all the wav files stored
            noise_path: path to noise signal
            snr: desired snr
            P: window length
            transform: func for feature generation
        '''

        self.x_path = x_path
        self.noise_path = noise_path
        self.transform = transform
        self.snr = snr
        self.P = P
        self.fnames = os.listdir(x_path)

    def __len__(self):
        return int(len(self.fnames))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fpath = os.path.join(self.x_path, self.fnames[idx])
        sample = self.transform(fpath, self.noise_path, self.snr, self.P)
        return sample, self.fnames[idx]


class Layer1(nn.Module):
    '''
    Train with mel features
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(704, 128)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        return self.out(x)

class Layer_1_2(nn.Module):
    def __init__(self, l1=None):
        super().__init__()
        if l1:
            self.fc1 = l1.fc1
        else:
            self.fc1 = nn.Linear(704, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 64)

    def forward(self, x):
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
        self.fc3 = nn.Linear(128, 128)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 64)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.out(x)
        return x 
        

def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predict, target, mask):
        predict = torch.flatten(predict)
        target = torch.flatten(target)
        mask = torch.flatten(mask)
        err = torch.sum(((predict-target)*mask)**2.0)/torch.sum(mask)
        return err

def pretrain(x_path, model_path, num_epochs, noise_path, snr, P, resume='False'):
    
    losses_l1 = []
    losses_l2 = []
    val_losses = []
    prev_val = 9999

    device = torch.device("cuda")

    ############# PRETRAIN FIRST LAYER ################

    if resume=='False':
    
        l1 = Layer1()
        l1 = l1.double()
        l1.apply(weights)
        criterion = MaskedMSELoss()
        optimizer = optim.SGD(l1.parameters(), lr=0.01, momentum=0.9)
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

            dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Train')

            loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
   
            for batch in loader:
                x = batch["x"]
                x = x.to(device)
                target = batch["t"]
                target = target.to(device)
                mask = batch["mask"].to(device)
                output = l1(x)
                newLoss = criterion(output, target, mask)              
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

                loss = newLoss.detach().cpu().numpy()
                epoch_loss+=loss

            losses_l1.append(epoch_loss/len(loader))
            np.save(model_path+"losses_l1.npy", np.asarray(losses_l1))
            print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))

            #### VALIDATION #####
    
            print('Starting validation...')
            
            dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Val')
            val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
            overall_val_loss=0

            for batch in val_loader:
                x = batch["x"]
                x = x.to(device)
                target = batch["t"]
                target = target.to(device)
                mask = batch["mask"].to(device)
                output = l1(x)
                valLoss = criterion(output, target, mask) 
                overall_val_loss+=valLoss.detach().cpu().numpy()

            curr_val_loss = overall_val_loss/len(val_loader)
            val_losses.append(curr_val_loss)
            print('Validation loss: ', curr_val_loss)
            np.save(model_path+'val_losses_l1.npy', np.asarray(val_losses))

            if curr_val_loss < prev_val:
                torch.save(best_l1, model_path+'dnn_map_l1_best.pth')
                prev_val = curr_val_loss
            torch.save(best_l1, model_path+"dnn_map_l1_last.pth")
    ###### TRAIN SECOND LAYER ##########
    prev_val=99999
    val_losses = []
    l1 = Layer1().double()

    l1.load_state_dict(torch.load(model_path+'dnn_map_l1_best.pth'))
    l1 = l1.to(device)

    l2 = Layer_1_2(l1)
    l2 = l2.double()
    criterion = MaskedMSELoss()
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

        dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Train')
        loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

        for batch in loader:
            x = batch["x"]
            x = x.to(device)
            target = batch["t"]
            target = target.to(device)
            mask = batch["mask"].to(device)
            output = l2(x)
            newLoss = criterion(output, target, mask)           
            optimizer.zero_grad()
            newLoss.backward()
            optimizer.step()
            
            epoch_loss+=newLoss.data.detach().cpu().numpy()

        losses_l2.append(epoch_loss/len(loader))
        np.save(model_path+"losses_l2.npy", np.asarray(losses_l2))
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))

        #### VALIDATION #####
       
        print('Starting validation...')
        dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Val')
        val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
        overall_val_loss=0

        for batch in val_loader:
            x = batch["x"]
            x = x.to(device)
            target = batch["t"]
            target = target.to(device)
            mask = batch["mask"].to(device)
            output = l2(x)
            valLoss = criterion(output, target, mask) 
            overall_val_loss+=valLoss.detach().cpu().numpy()

        curr_val_loss = overall_val_loss/len(val_loader)
        val_losses.append(curr_val_loss)
        print('Validation loss: ', curr_val_loss)
        np.save(model_path+'val_losses_l2.npy', np.asarray(val_losses))

        if curr_val_loss < prev_val:
            torch.save(best_l2, model_path+'dnn_map_l2_best.pth')
            prev_val = curr_val_loss
        torch.save(best_l2, model_path+"dnn_map_l2_last.pth")

            


def train_dnn(x_path, model_path, num_epochs, noise_path, snr, P, 
             from_pretrained='True', resume='False'):
    
    if from_pretrained=='True':
        print("Loading pretrained weights...")
        l1 = Layer1()
        l1.load_state_dict(torch.load(model_path+'dnn_map_l1_best.pth'))
        l1_2 = Layer_1_2(l1)
        l1_2.load_state_dict(torch.load(model_path+'dnn_map_l2_best.pth'))
        model = DNN_mel(l1_2).double()

    elif resume=="True":
        model = DNN_mel()
        model.load_state_dict(torch.load(model_path+'dnn_map_best.pth')).double()

    criterion = MaskedMSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda:3")
    model.cuda()
    model = model.to(device)
    criterion.cuda()


    #Training loop

    best_model = copy.deepcopy(model.state_dict())
    losses = []
    val_losses = []
    
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0
        
        dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Train')
        loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
        

        for batch in loader:
            x = batch["x"]
            x = x.to(device)
            target = batch["t"]
            target = target.to(device)
            mask = batch["mask"].to(device)
            output = model(x)
            newLoss = criterion(output, target, mask)
            epoch_loss+=newLoss.detach().cpu().numpy()           
            optimizer.zero_grad()
            newLoss.backward()
            optimizer.step()

        losses.append(epoch_loss/len(loader))
        np.save(model_path+"losses.npy", losses)
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/len(loader)))

        #### VALIDATION #####
       
        print('Starting validation...')
        prev_val = 9999

        dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Val')
        val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
        overall_val_loss=0

        for batch in val_loader:
            x = batch["x"]
            x = x.to(device)
            target = batch["t"]
            target = target.to(device)
            mask = batch["mask"].to(device)
            output = model(x)
            valLoss = criterion(output, target, mask) 
            overall_val_loss+=valLoss.detach().cpu().numpy()

        curr_val_loss = overall_val_loss/len(val_loader)
        val_losses.append(curr_val_loss)
        print('Validation loss: ', curr_val_loss)
        np.save(model_path+'val_losses.npy', np.asarray(val_losses))

        if curr_val_loss < prev_val:
            torch.save(best_model, model_path+'dnn_map_best.pth')
            prev_val = curr_val_loss
        torch.save(best_model, model_path+"dnn_map_last.pth")



def dnn_predict(x_path, noise_path, model_path, out_path, snr=0, P=5):
    device = torch.device("cuda")
    model = DNN_mel()
    model.load_state_dict(torch.load(os.path.join(model_path, 'dnn_map_best.pth')))
    model.cuda()
    model = model.to(device)
    model = model.double()

    dataset = DnnTestLoader(x_path, noise_path, snr, P, make_dnn_feats)
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("Predicting outputs...")
    num_steps = len(loader)
    step = 0
    for batch, fnames in loader:
        step+=1
        print('Step:{:4}/{:4}'.format(step, num_steps))
        x = batch["x"]
        x = x.to(device)
        masks = batch["mask"]
        output = model(x)
        print("out:", output.shape)
        for i, ex in enumerate(output):
            mask = masks[i]
            pad_ind = torch.sum(mask, dim=1).detach().cpu().numpy()[-1]
            print("mask:", mask.shape, pad_ind)
            ex = np.exp(ex.detach().cpu().numpy())[:, pad_ind]
            fname = fnames[i]
            np.save(os.path.join(out_path, fname), ex)
            
