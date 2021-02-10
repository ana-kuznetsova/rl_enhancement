import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os
import copy
import soundfile as sf

from pypesq import pesq
from tqdm import tqdm
import librosa



def generate_curriculum(clean_path, noisy_path, model_path):
    fnames = os.listdir(clean_path)
    d = {"fname":[], "pesq":[]}
    for fname in tqdm(fnames):
        wav_noisy, sr = librosa.core.load(os.path.join(noisy_path, fname), sr=16000)
        wav_clean, sr = librosa.core.load(os.path.join(clean_path, fname), sr=16000)
        score = pesq(wav_clean, wav_noisy)
        d["fname"].append(fname)
        d["pesq"].append(score)
    df = pd.DataFrame.from_dict(d)
    df = df.sort_values(by=['pesq'])
    df.to_csv(os.path.join(model_path, "train_sort.tsv"), sep='\t')


generate_curriculum('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/', 
                '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/',
                '/nobackup/anakuzne/data/experiments/speech_enhancement/curriculum/')
        