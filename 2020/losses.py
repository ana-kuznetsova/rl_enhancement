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

    def calc_true_pesq(self, x, y, s, mask, fs=16000):
        scores = []
        for i in range(x.shape[0]):
            ind = int(torch.sum(mask[i], 1))
            x = x[i][:, :ind]
            y = y[i][:, :ind]
            s = s[i][:, :ind]

            x = torch.istft(x, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True).detach().cpu().numpy()
           
           
            y = torch.istft(y, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True).detach().cpu().numpy()
            s = torch.istft(s, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True).detach().cpu().numpy()
            score_x = pesq(s, x, fs)
            score_y = pesq(s, y, fs)
            score_s = pesq(s, s, fs)
            scores.append([score_x, score_y, score_s]) 
        return scores
           

    def forward(self, x, y, s, mask, pred_scores, device):
        '''
        Args:
            x (batch): stfts of noisy speech
            y (batch): stfts of predicted speech from Actor
            s (batch): stfts of target speech
            pred_scores (batch): [x, y, s] predicted by Critic scored for x, y, s
            mask (batch): masks for signals
        '''
        true_scores = self.calc_true_pesq(x, y, s, mask)
        print(true_scores)
        final_score = []

        for i in range(x.shape[0]):
            e_x = (true_scores[i][0]-pred_scores[i][0])**2
            e_y = (true_scores[i][1]-pred_scores[i][1])**2
            e_s = (true_scores[i][2]-pred_scores[i][2])**2
            final_score.append(e_x + e_y + e_s)
        res = sum(final_score)
        res = torch.tensor(res, requires_grad=True).to(device)
        return res
            