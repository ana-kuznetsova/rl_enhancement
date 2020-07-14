import torch
import torch.nn.functional as Func
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import copy
import pickle

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
    #Change if not enough memory
    X_chunk, y_chunk = make_batch(x_path, y_path, [0, chunk_size], 5, maxlen, win_len, hop_size, fs)

    trainData = data.DataLoader(trainDataLoader(X_chunk, y_chunk), batch_size = 64)


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        loss = 0.0 
        for step, (audio, target) in enumerate(trainData): 
            audio = audio.to(device)
            target = target.to(device)
            model.train()
            output = model(audio)
            newLoss = criterion(output,target)
            loss += newLoss.data
            optimizer.zero_grad()
            newLoss.backward()
            optimizer.step()
            #print('Chunk:{:2} Training loss:{:>4f}'.format(chunk+1, chunk_loss))
            #loss += chunk_loss
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


def inference(test_data_path, out_test,
             win_len=512, hop_size=256, fs=44000):
    model = DNN()
    model.load_state_dict(torch.load(model_path+'dnn_map_best.pth'))
    testData = data.DataLoader(testDataLoader())

    names = testData.test_files

    for step, audio in enumerate(testData):
        print(step)
    

    name = names[step].split('.')[0]
    name = name+'.wav'    
    model.eval()
    output = model(audio)
    output = librosa.istft(np.transpose(output[0].cpu().data.numpy().squeeze()), hop_length=hop_size,
                            win_length=win_len) 
    librosa.output.write_wav(out_test+name, output, fs) 