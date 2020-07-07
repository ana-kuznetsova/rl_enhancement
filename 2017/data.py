import os
from tqdm import tqdm
import librosa
import numpy as np

from utils import read
from utils import write_npy
from utils import collect_paths


def STFT(x, win_len, hop_size, win_type='hann'):
    return librosa.core.stft(x, win_len, hop_size, win_len, win_type)

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