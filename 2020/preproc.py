import torchaudio
import torch
import torch.utils.data as data
import torch.nn as nn


import os
import numpy as np
import librosa
from tqdm import tqdm

def find_max(path):
    '''
    Voice-DEMAND, 28 spk: 1890  maxlen
    '''
    fnames = os.listdir(path)
    max_len = 0
    stft = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=512, hop_length=128)
    for f in tqdm(fnames):
        speech, sr = librosa.core.load(os.path.join(path, f), sr=16000)
        speech = stft(torch.tensor(speech))
        if speech.shape[1] > max_len:
            max_len = speech.shape[1]
    
    print("Maximum input length:", max_len)

def get_feats(clean_path, noisy_path, maxlen=1890):
    clean, sr = librosa.core.load(clean_path, sr=16000)
    noisy, sr = librosa.core.load(noisy_path, sr=16000)

    #stft = torch.stft(n_fft=1024, win_length=512, hop_length=128, return_complex=True)
    clean = torch.stft(torch.tensor(clean), n_fft=1024, win_length=512, hop_length=128, return_complex=True)
    noisy = torch.stft(torch.tensor(noisy), n_fft=1024, win_length=512, hop_length=128, return_complex=True)
    
    mask = torch.ones(1, clean.shape[1])
    mask = nn.ZeroPad2d(padding=(0, maxlen-clean.shape[1], 0, 0))(mask)
    clean = nn.ZeroPad2d(padding=(0, maxlen-clean.shape[1], 0, 0))(clean)
    noisy = nn.ZeroPad2d(padding=(0, maxlen-noisy.shape[1], 0, 0))(noisy)
    return {"clean":clean, "noisy":noisy, "mask":mask}

class DataLoader(data.Dataset):
    def __init__(self, clean_path, noisy_path, transform):
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.fnames_clean = self.get_samples()[0]
        self.fnames_noisy = self.get_samples()[1]
        self.transform = transform

    def get_samples(self):
        fnames = os.listdir(self.clean_path)
        fnames = np.random.choice(fnames, size=1000, replace=True)
        clean = [os.path.join(self.clean_path, n) for n in fnames]
        noisy = [os.path.join(self.noisy_path, n) for n in fnames]
        return (clean, noisy)

    def __len__(self):
        return len(self.fnames_clean)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.transform(self.fnames_clean[idx], self.fnames_noisy[idx])
        return sample