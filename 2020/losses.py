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
            print(x.shape)

            x = torch.istft(x, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True)
            print(x.shape)

            #print(x.shape, y.shape, s.shape)
            '''
            
            y = torch.stft(y, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True, return_complex=True)
            s = torch.stft(s, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True, return_complex=True)
            score_x = pesq(s, x, fs)
            score_y = pesq(s, y, fs)
            score_s = pesq(s, s, fs)
            #print(score_s, score_x, score_y)
            score_s.append([score_x, score_y, score_s]) 
            return scores
            '''

    def forward(self, x, y, s, mask, pred_scores):
        '''
        Args:
            x (batch): stfts of noisy speech
            y (batch): stfts of predicted speech from Actor
            s (batch): stfts of target speech
            pred_scores (batch): [x, y, s] predicted by Critic scored for x, y, s
            mask (batch): masks for signals
        '''
        final_score = None

        for i in range(x.shape[0]):
            true_scores = self.calc_true_pesq(x, y, s, mask)
            print(true_scores)