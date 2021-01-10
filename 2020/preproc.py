import torchaudio
import torch
import torch.utils.data as data


import os
import numpy as np

def get_feats(clean_path, noisy_path):
    clean = torchaudio.load_wav(clean_path, sample_rate=16000)
    noisy = torchaudio.load_wav(noisy_path, sample_rate=16000)

    stft = torchaudio.transforms.Spectrogram(win_length=512, hop_length=128)
    clean = stft(clean)
    noisy = stft(noisy)
    return {"clean":clean, "noisy":noisy}


def get_samples(num_samples, clean_path, noisy_path):
    pass

class DataLoader(data.Dataset):
    def __init__(self, fnames_clean, fnames_noisy, transform):
        self.fnames_clean = fnames_clean
        self.fnames_noisy = fnames_noisy
        self.transforms = transform
    
    def __len__(self):
        return len(self.fnames_clean)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.transform(self.fnames_clean, self.fnames_noisy)
        return sample