import torch
import torch.nn as nn
import torchaudio

from pypesq import pesq

class SDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def clip(self, val, alpha=20):
        return alpha*torch.tanh(val/alpha)


    def forward(self, x, t, y):
        temp = []
        for i in range(len(t)):
            if torch.any(torch.isnan(t[i])) or torch.any(torch.isnan(y[i])):
                print(t[i], y[i])
            #print("INPUT:", x[i][:5], t[i][:5], y[i][:5])
            #print("Norms:", torch.norm(t[i]),  torch.norm(t[i]-y[i]))
            frac1 = torch.norm(t[i])/torch.norm((t[i]-y[i]))
            #print("FRACS1:", frac1)
            frac1 = -0.5*self.clip(10*torch.log10(frac1))
            print("FRAC1:", frac1)

            n = x[i]-t[i]
            diff = n - (x[i]-y[i])
            frac2 = torch.norm(n)/torch.norm(diff)
            print("FRAC2:", frac2)
            frac2 = 0.5*self.clip(10*torch.log10(frac2))
            val = frac1-frac2
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
            x_i = x[i][:, :ind]
            y_i = y[i][:, :ind]
            s_i = s[i][:, :ind]

            x_i = torch.istft(x_i, n_fft=512, win_length=512, hop_length=128, normalized=True).detach().cpu().numpy()
           
           
            y_i = torch.istft(y_i, n_fft=512, win_length=512, hop_length=128, normalized=True).detach().cpu().numpy()
            s_i = torch.istft(s_i, n_fft=512, win_length=512, hop_length=128, normalized=True).detach().cpu().numpy()
            score_x = pesq(s_i, x_i, fs)
            score_y = pesq(s_i, y_i, fs)
            score_s = pesq(s_i, s_i, fs)
            del x_i
            del y_i
            del s_i
            scores.append([score_x, score_y, score_s]) 
        return torch.tensor(scores)
           

    def forward(self, x, y, s, mask, pred_scores, device):
        '''
        Args:
            x (batch): stfts of noisy speech
            y (batch): stfts of predicted speech from Actor
            s (batch): stfts of target speech
            pred_scores (batch): [x, y, s] predicted by Critic scored for x, y, s
            mask (batch): masks for signals
        '''
        true_scores = self.calc_true_pesq(x, y, s, mask).to(device)
        temp = torch.sum((true_scores - pred_scores)**2, 0)/true_scores.shape[0]
        res = torch.sum(temp)
        return res
            

class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds):
        return -torch.sum(preds)