import torch
import torch.nn as nn
import torchaudio

from pypesq import pesq

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
            val = self.clip(val)
            temp.append(val)
        temp = torch.stack(temp)
        return torch.sum(temp)

class CriticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, s, pred_scores, mask):
        '''
        Args:
            x (batch): stfts of noisy speech
            y (batch): stfts of predicted speech from Actor
            s (batch): stfts of target speech
            pred_scores: [x, y, s] predicted by Critic scored for x, y, s
            mask (batch): masks for signals
        '''
        final_score = None

        for i in range(x.shape[0]):
            pass