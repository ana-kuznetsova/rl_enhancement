import librosa
import os
import numpy as np

def read(path, fs):
    file_, fs = librosa.core.load(path, fs, mono=True)
    return file_[:-1]

def write(file_name, signal, fs):
    librosa.output.write_wav(file_name, signal, fs)


def collect_paths(train_dir):
    paths = []
    for dirs, subdirs, files in os.walk(train_dir):
      for file in files:
        if '.wav' in file:
          paths.append(os.path.join(dirs, file))
    return paths

def write_npy(path, filename, array):
    np.save(os.path.join(path, filename), array)

def pad(vec, maxlen):
    if vec.shape[1] == maxlen:
        return vec
    return np.pad(vec, ((0, 0), (0, maxlen-vec.shape[1])), 'constant')


def make_csv(train_noisy_path, irm_path, df_path):
    irms_ordered = []
    train_files = os.listdir(train_noisy_path)
    irms = os.listdir(irm_path)
    for t in train_files:
        i = irms.index(t)
        irms_ordered.append(irm_path+irms[i])
    train_files = [train_noisy_path+t  for t in train_files]
    train_df = pd.DataFrame({"train_path":train_files, "irm_path":irms_ordered})
    train_df.to_csv(df_path+'train_paths.csv')