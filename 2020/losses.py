import torch
import torch.nn as nn
import torchaudio

from pypesq import pesq

class SDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def clip(self, val, alpha=20):
        return alpha*torch.tanh(val/alpha)


    def forward(self, t, y):
        temp = []
        for i in range(len(t)):
            if torch.isnan(t[i]).any():
                print(t[i])
            frac = torch.norm(t[i])/torch.norm((t[i]-y[i]))
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
            x_i = x[i][:, :ind]
            y_i = y[i][:, :ind]
            s_i = s[i][:, :ind]

            x_i = torch.istft(x_i, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True).detach().cpu().numpy()
           
           
            y_i = torch.istft(y_i, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True).detach().cpu().numpy()
            s_i = torch.istft(s_i, n_fft=1024, win_length=512, hop_length=128, 
                           normalized=True).detach().cpu().numpy()
            score_x = pesq(s_i, x_i, fs)
            score_y = pesq(s_i, y_i, fs)
            score_s = pesq(s_i, s_i, fs)
            del x_i
            del y_i
            del s_i
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
        print("Pesq:", true_scores)
        final_score = []

        for i in range(x.shape[0]):
            e_x = (true_scores[i][0]-pred_scores[0][i])**2
            e_y = (true_scores[i][1]-pred_scores[1][i])**2
            e_s = (true_scores[i][2]-pred_scores[2][i])**2
            final_score.append(e_x + e_y + e_s)
            del e_x
            del e_y
            del e_s
        res = sum(final_score)
        res = torch.tensor(res, requires_grad=True).to(device)
        return res
            