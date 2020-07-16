import os
from tqdm import tqdm
import librosa
import numpy as np

from utils import read
from utils import write_npy
from utils import collect_paths
from utils import pad

def generate_noisy(speech, noise, desired_snr):    
    #calculate energies
    E_speech = np.sum(np.power(speech, 2))
    E_noise = np.sum(np.power(noise, 2))
    
    #calculate b coeff
    b = np.sqrt((E_speech/(np.power(10, (desired_snr/10))))/E_noise)    
    return speech + b*noise


def create_noisy_data(data_paths, out_path, noise_path, 
                      target_SNR, win_len, hop_size, fs):
    noise = read(noise_path, fs)

    for path in tqdm_notebook(data_paths):
        speech = read(path, fs)
        noise = pad_noise(speech, noise)
        blend = generate_noisy(speech, noise, target_SNR)
        stft = STFT(blend, win_len, hop_size)
        fname = path.split('/')[-2]+ '_' + path.split('/')[-1].split('.')[0] + '.npy'
        np.save(out_path+fname, stft)


def STFT(x, win_len, hop_size, win_type='hann'):
    return librosa.core.stft(x, win_len, hop_size, win_len, win_type)


def mel_spec(stft, win_len, hop_size, fs):
    mel_spec = librosa.feature.melspectrogram(sr=fs, S=stft,
                                              n_fft=win_len,
                                              hop_length=hop_size,
                                              n_mels=64)
    return mel_spec

def calc_SNR(speech, noise):        
    E_speech = np.power(speech, 2)
    E_noise = np.power(noise, 2)
    return 10*np.log10(E_speech/E_noise)

def convert_SNR(speech, noise, target_snr):
    '''
    speech: wav input
    noise wav input
    '''
    E_speech = np.power(speech, 2)
    E_noise = np.power(noise, 2)  
    b = np.sqrt((E_speech/(np.power(10, (target_snr/10))))/E_noise)
    target = speech + b*noise
    
    if np.argwhere(np.isnan(target)):
        target = np.nan_to_num(target)
    return target

def IRM(speech, noise):
    '''
    Params:
        speech: stft matrix of speech signal
        noise: stft matrix of noise signal
    '''
    mask = np.power(10, calc_SNR(speech, noise)/10)/(np.power(10, calc_SNR(speech, noise)/10) + 1)
    return mask

def Wiener(speech, noise):
    f = (np.abs(speech)**2)/((np.abs(speech)**2)/np.abs(noise)**2)
    return f


def pad_noise(speech, noise):
    '''
    Cuts noise vector if speech vec is shorter
    Adds noise if speech vector is longer
    '''
    noise_len = noise.shape[0]
    speech_len = speech.shape[0]
    
    if speech_len > noise_len:
        repeat = speech_len//noise_len
        if repeat == 1:
            diff = speech_len - noise_len
            noise = np.concatenate((noise, noise[:diff]), axis=0)
        else:
            noise = np.tile(noise, repeat)
            diff = speech_len - noise.shape[0]
            noise = np.concatenate((noise, noise[:diff]))           
            
    elif speech_len < noise_len:
        noise = noise[:speech_len]  
    return noise


def calc_masks(speech_paths, noise_path, fs, win_len, hop_size,
               stft_dir,
               mask_dir,
               mask_type='IRM',
               write_stft=False):
    
    noise = read(noise_path, fs)

    for p in tqdm(speech_paths):
        fname = p.split('/')[-1].split('.')[0] + '.npy'
        speech = read(p, fs)
        noise = pad_noise(speech, noise)
        stft_noise = STFT(noise, win_len, hop_size)
        stft_clean = STFT(speech, win_len, hop_size)
        if write_stft:
            write_npy(stft_dir, fname, stft_clean)
        if mask_type=='IRM':
            irm = IRM(stft_clean, stft_noise)
            write_npy(mask_dir, fname, irm)
        elif mask_type=='Wiener':
            wiener = Wiener(stft_clean, stft_noise)
            write_npy(mask_dir, fname, wiener)


def get_X_batch(stft, P):
    windows = []
    for col in range(-P, stft.shape[1]-P):
        win = np.zeros((stft.shape[0], (2*P)+1))
        if col < 0:
            win[:,-col:] = stft[:,:P+1+(P+col)]    
        
        if col >= 0 and col < stft.shape[1] - 2*P:
            win = stft[:, col:col+2*P+1]
        
        if col >= stft.shape[1] - 2*P:
            temp = stft[:,col:]
            win[:,:temp.shape[1]] = stft[:,col:]
        windows.append(win.flatten('F'))
        
    return np.asarray(windows)


def make_batch(x_path, y_path, ind, P, maxlen, win_len, hop_size, fs):
    X = []
    y = []
    chunk_x = os.listdir(x_path)[ind[0]:ind[1]]

    for path in tqdm(chunk_x):
        arr = np.load(x_path+path)
        arr = np.abs(get_X_batch(arr, P)).T
        arr = pad(arr, maxlen)
        X.extend(arr.T)

        arr = np.load(y_path+path)
        arr = np.abs(pad(arr, maxlen))
        y.extend(arr)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y 


def make_batch_test(x_path, ind, P, maxlen=1339, win_len=512, hop_size=256, fs=44000):
    X = []
    chunk_x = os.listdir(x_path)[ind[0]:ind[1]]
    for path in tqdm(chunk_x):
        arr = np.load(x_path+path)
        arr = get_X_batch(arr, P)
        arr = np.abs(mel_spec(arr, win_len, hop_size, fs))
        X.extend(arr)
    X = np.asarray(X)
    return X