import torch
import torch.nn as nn
import torchaudio

class SDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def norm2(self, vec):
        return torch.sqrt(torch.sum(vec**2))
    
    def clip(self, val, alpha=20):
        return alpha*torch.tanh(val/alpha)


    def forward(self, t, y):
        temp = []
        for i in range(len(t)):
            frac = self.norm2(t[i])/self.norm2((t[i]-y[i]))
            val = 10*torch.log10(frac)
            print(val)
            val = self.clip(val)
            print(val)
            temp.append(val)
        temp = torch.stack(temp)
        return torch.sum(temp)