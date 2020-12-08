import librosa
import os
import numpy as np

def read(path, fs=16000):
    file_, fs = librosa.core.load(path, fs, mono=True)
    return file_

def pad(vec, maxlen):
    if vec.shape[1] == maxlen:
        return vec
    return np.pad(vec, ((0, 0), (0, maxlen-vec.shape[1])), 'constant')