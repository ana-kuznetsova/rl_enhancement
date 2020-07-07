import librosa
import os
import numpy as np

def read(path, sr):
    file, sr = librosa.core.load(path, sr, mono=True)
    return file[:-1]

def write(file_name, signal):
    librosa.output.write_wav(file_name, signal, 16000)


def collect_paths(train_dir):
    paths = []
    for dirs, subdirs, files in os.walk(train_dir):
        if len(dirs.split('/')) == 12:
            for file in files:
                if '.wav' in file:
                    paths.append(os.path.join(dirs, file))
    return paths

def write_npy(path, filename, array):
    np.save(os.path.join(path, filename), array)
