import numpy as np
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


from preproc import make_dnn_feats


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
        self.out = nn.Linear(128, 257)

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
        self.out = nn.Linear(128, 257)
        
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


def pretrain(x_path, model_path, num_epochs, noise_path, snr, P, resume='False'):
    
    losses_l1 = []
    losses_l2 = []
    val_losses = []
    prev_val = 9999

    ############# PRETRAIN FIRST LAYER ################
    if resume=='False':
    
        l1 = Layer1()
        #l1 = l1.double()
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

            dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Train')

            loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
            for sample in loader:
                #print('x:', sample["x"].shape, "t", sample["t"].shape)
                x = x.to(device)
                print("X:", x.shape)
                target = target.to(device)
                print("t:", target.shape)
                output = l1(x)
                print("output:", output)

                newLoss = criterion(output, target)              
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

            loss = newLoss.detach().cpu().numpy()
            epoch_loss+=loss

            losses_l1.append(epoch_loss/epoch)
            np.save(model_path+"losses_l1.npy", np.asarray(losses_l1))
            print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/epoch))

            #### VALIDATION #####
        
            print('Starting validation...')
            
            dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Val')
            val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
            overall_val_loss=0

            for x, target in val_loader:
                x = x.to(device)
                target = target.to(device).float()
                output = l1(x)
                valLoss = criterion(output, target)
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
    l1 = Layer1()

    l1.load_state_dict(torch.load(model_path+'dnn_map_l1_last.pth'))

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

        dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Train')
        loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

        for x, target in loader:
            x = x.to(device)
            target = target.to(device)
            output = l2(x)
            newLoss = criterion(output, target)              
            optimizer.zero_grad()
            newLoss.backward()
            optimizer.step()

            epoch_loss+=newLoss.data.detach().cpu().numpy()

        losses_l2.append(epoch_loss/epoch)
        np.save(model_path+"losses_l2.npy", np.asarray(losses_l2))
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/epoch))

        #### VALIDATION #####
       
        print('Starting validation...')
        dataset = DnnLoader(x_path, noise_path, snr, P, make_dnn_feats, mode='Val')
        val_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
        overall_val_loss=0

        for x, target in val_loader:
            x = x.to(device)
            target = target.to(device).float()
            output = l2(x)
            valLoss = criterion(output, target)
            overall_val_loss+=valLoss.detach().cpu().numpy()

        curr_val_loss = overall_val_loss/len(val_loader)
        val_losses.append(curr_val_loss)
        print('Validation loss: ', curr_val_loss)
        np.save(model_path+'val_losses_l2.npy', np.asarray(val_losses))

        if curr_val_loss < prev_val:
            torch.save(best_l2, model_path+'dnn_map_l2_best.pth')
            prev_val = curr_val_loss
        torch.save(best_l2, model_path+"dnn_map_l2_last.pth")

            


def train_dnn(chunk_size,
              model_path, x_path, y_path,  pretrain_path, from_pretrained='False',
              maxlen=1339, win_len=512, hop_size=256, fs=16000, resume='False'):
    
    num_epochs = 50
    
    if from_pretrained=='True':
        print("Loading pretrained weights...")
        l1 = Layer1()
        l1.load_state_dict(torch.load(pretrain_path+'dnn_map_l1_best.pth'))
        l1_2 = Layer_1_2(l1)
        l1_2.load_state_dict(torch.load(pretrain_path+'dnn_map_l2_best.pth'))
        model = DNN_mel(l1_2)

    elif resume=="True":
        model = DNN_mel()
        model.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))

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
    val_losses = []
    
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss = 0
        
        num_chunk = (12474//chunk_size) + 1
        for chunk in range(num_chunk):
            chunk_loss = 0
            start = chunk*chunk_size
            end = min(start+chunk_size, 12474)
            print(start, end)

            X_chunk, y_chunk, batch_indices = make_windows(x_path, y_path,
                                          [start, end], P=5, 
                                           win_len=512, 
                                           hop_size=256, fs=16000, nn_type='map')

            dataset = QDataSet(X_chunk, y_chunk, batch_indices)
            loader = data.DataLoader(dataset, batch_size=1)

            for x, target in loader:
                x = x.to(device)
                x = x.reshape(x.shape[1], x.shape[2])
                target = target.to(device).float()
                target = target.reshape(target.shape[1], target.shape[2])
                output = model(x)

                newLoss = criterion(output, target)              
                chunk_loss += newLoss.data
                optimizer.zero_grad()
                newLoss.backward()
                optimizer.step()

            chunk_loss = (chunk_loss.detach().cpu().numpy())/len(loader)
            
            epoch_loss+=chunk_loss

            print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))

        losses.append(epoch_loss/num_chunk)
        np.save(model_path+"losses.npy", losses)
        print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, epoch_loss/num_chunk))

        #### VALIDATION #####
       
        print('Starting validation...')
        prev_val = 9999
        start = 12474
        end = 13860
        X_val, A_val, batch_indices = make_windows(x_path, y_path,
                                            [start, end], P=5, 
                                            win_len=512, 
                                            hop_size=256, fs=16000, nn_type='map')

        dataset = QDataSet(X_val, A_val, batch_indices)
        val_loader = data.DataLoader(dataset, batch_size=1)
        overall_val_loss=0

        for x, target in val_loader:
            x = x.to(device)
            x = x.reshape(x.shape[1], x.shape[2])
            target = target.to(device).float()
            target = target.reshape(target.shape[1], target.shape[2])
            output = model(x)
            valLoss = criterion(output, target)
            overall_val_loss+=valLoss.detach().cpu().numpy()

        curr_val_loss = overall_val_loss/len(val_loader)
        val_losses.append(curr_val_loss)
        print('Validation loss: ', curr_val_loss)
        np.save(model_path+'val_losses.npy', np.asarray(val_losses))

        if curr_val_loss < prev_val:
            torch.save(best_model, model_path+'dnn_map_best.pth')
            prev_val = curr_val_loss
        torch.save(best_model, model_path+"dnn_map_last.pth")



def inference(chunk_size, x_path, y_path, model_path,
              test_out,
              win_len=512, hop_size=256, fs=16000):

    
    device = torch.device("cuda")
    model = DNN_mel()
    model.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))
    model.cuda()
    model = model.to(device)

    fnames = os.listdir(x_path)

    X_test, y_test, batch_indices = make_windows(x_path, y_path,
                                            [0, len(fnames)], P=5, 
                                            win_len=512, 
                                            hop_size=256, fs=16000, nn_type='map')

    dataset = QDataSet(X_test, y_test, batch_indices)
    test_loader = data.DataLoader(dataset, batch_size=1)

    for i, (x, target) in enumerate(test_loader):
        x = x.to(device)
        x = x.reshape(x.shape[1], x.shape[2])
        target = target.to(device).float()
        target = target.reshape(target.shape[1], target.shape[2])
        output = model(x).cpu().data.numpy().T
        np.save(test_out+fnames[i], output)
