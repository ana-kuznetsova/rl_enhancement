import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from preproc import DataLoader
from preproc import get_feats

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=30,
                                 kernel_size=(5, 15), stride=(1,1), 
                                 padding=(2,7))
        self.conv2d2 = nn.Conv2d(in_channels=30, out_channels=60,
                                 kernel_size=(5, 15), stride=(1,1), 
                                 padding=(2,7))
        self.conv2d3 = nn.Conv2d(in_channels=60, out_channels=1,
                                 kernel_size=(1, 1), stride=(1,1))
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(513, 128)
        self.bi_lstm = nn.LSTM(128, hidden_size=256, num_layers=2, 
                               batch_first=True, dropout=0.3, bidirectional=True)
        self.linear2 = nn.Linear(512, 2*513)

    def forward(self, x):
        x = self.relu(self.conv2d1(x))
        x = self.relu(self.conv2d2(x))
        x = self.relu(self.conv2d3(x))
        x = x.squeeze(1)
        
        x_batch = []

        ##Run through linear layer
        for i in range(x.shape[0]):
            curr_x = self.linear1(x[i].T)
            x_batch.append(curr_x)
        x = torch.stack(x_batch)
        del x_batch
    
        x, (h, _) = self.bi_lstm(x)

        x_batch = []
        for i in range(x.shape[0]):
            curr_x = self.linear2(x[i])
            x_batch.append(curr_x)

        real = []
        imag = []
        
        for m in x_batch:
            r = m[:int(m.shape[1]/2),:]
            real.append(r)
            i = m[int(m.shape[1]/2):,:]
            imag.append(i)

        return torch.stack(real), torch.stack(imag)

device = torch.device("cuda:1")
model = Actor()
model.cuda()
model = model.to(device)

dataset = DataLoader('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
                     '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', get_feats)
loader = data.DataLoader(dataset, batch_size=10, shuffle=True)

for batch in loader:
    x = batch["noisy"].unsqueeze(1).to(device)
    t = batch["clean"].unsqueeze(1).to(device)
    out_r, out_i = model(x)
    print(out_r.shape, out_i.shape)