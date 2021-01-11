import torchaudio
import torch
import torch.utils.data as data


import os
import numpy as np
import librosa

def get_feats(clean_path, noisy_path):
    clean, sr = librosa.core.load(clean_path, sr=16000)
    noisy, sr = librosa.core.load(noisy_path, sr=16000)

    stft = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=512, hop_length=128)
    clean = stft(torch.tensor(clean))
    noisy = stft(torch.tensor(noisy))
    return {"clean":clean, "noisy":noisy}

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