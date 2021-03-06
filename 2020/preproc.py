import torchaudio
import torch
import torch.utils.data as data
import torch.nn as nn


import os
import numpy as np
from tqdm import tqdm
import csv
import soundfile as sfl
import librosa


def collate_custom(data):
    '''
    For batch
    '''
    def maxlen_fn(paths):
        max_len=0
        for f in paths:
            sig, sr = librosa.core.load(f)
            sig = librosa.stft(sig, n_fft=512, hop_length=128, win_length=512)
            if sig.shape[1] > max_len:
                max_len = sig.shape[1]
        return int(max_len)

    clean_paths = [ex[0] for ex in data]
    noisy_paths = [ex[1] for ex in data]

    maxlen = maxlen_fn(clean_paths)
    batch_clean = []
    batch_noisy = []
    batch_mask = []
    
    for clean, noisy in zip(clean_paths, noisy_paths):
        clean, sr = librosa.core.load(clean, sr=16000)
        noisy, sr = librosa.core.load(noisy, sr=16000)
        #clean = torch.stft(torch.tensor(clean).float(), n_fft=512, win_length=512, hop_length=128, return_complex=True, normalized=True)
        #noisy = torch.stft(torch.tensor(noisy).float(), n_fft=512, win_length=512, hop_length=128, return_complex=True, normalized=True)
        clean = torch.tensor(librosa.stft(clean, n_fft=512, hop_length=128, win_length=512))
        noisy = torch.tensor(librosa.stft(noisy, n_fft=512, hop_length=128, win_length=512))
        mask = torch.ones(1, clean.shape[1])
        mask = nn.ZeroPad2d(padding=(0, maxlen-clean.shape[1], 0, 0))(mask)
        clean = nn.ZeroPad2d(padding=(0, maxlen-clean.shape[1], 0, 0))(clean)
        noisy = nn.ZeroPad2d(padding=(0, maxlen-noisy.shape[1], 0, 0))(noisy)
        batch_clean.append(clean)
        batch_noisy.append(noisy)
        batch_mask.append(mask)
    return {"clean":torch.stack(batch_clean), "noisy":torch.stack(batch_noisy), "mask":torch.stack(batch_mask)}



class Data(data.Dataset):
    def __init__(self, clean_path, noisy_path, sample_size=1000):
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.sample_size = sample_size
        self.fnames = self.get_samples()
        self.fnames_clean = self.fnames[0]
        self.fnames_noisy = self.fnames[1]

    def get_samples(self):
        lengths = []
        fnames = os.listdir(self.clean_path)
        fnames = np.random.choice(fnames, size=self.sample_size, replace=True)
        
        #print("Bucketing in progress...")
        for p in fnames:
            p = os.path.join(self.noisy_path, p)
            size = os.stat(p).st_size
            lengths.append((p, size))
        lengths = sorted(lengths, key=lambda x:x[1])

        clean = [os.path.join(self.clean_path, n[0]) for n in lengths]
        noisy = [os.path.join(self.noisy_path, n[0]) for n in lengths]
        return (clean, noisy)

    def __len__(self):
        return len(self.fnames_clean)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.fnames_clean[idx], self.fnames_noisy[idx])
        return sample


class DataTest(data.Dataset):
    def __init__(self, clean_path, noisy_path):
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.fnames = self.get_samples()
        self.fnames_clean = self.fnames[0]
        self.fnames_noisy = self.fnames[1]

    def get_samples(self):
        fnames = os.listdir(self.clean_path)
        clean = [os.path.join(self.clean_path, n) for n in fnames]
        noisy = [os.path.join(self.noisy_path, n) for n in fnames]
        return (clean, noisy)

    def __len__(self):
        return len(self.fnames_clean)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.fnames_clean[idx], self.fnames_noisy[idx])
        return sample