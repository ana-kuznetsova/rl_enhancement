import os
import librosa
import numpy as np
from utils import read





def pad_inf(vec, maxlen):
    if vec.shape[1] == maxlen:
        return vec
    return np.pad(vec, ((0, 0), (0, maxlen-vec.shape[1])), 'constant', constant_values=(-inf))

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

def generate_noisy(speech, noise, desired_snr):    
    #calculate energies
    E_speech = np.sum(np.power(speech, 2))
    E_noise = np.sum(np.power(noise, 2))
    
    #calculate b coeff
    b = np.sqrt((E_speech/(np.power(10, (desired_snr/10))))/E_noise)    
    return speech + b*noise


def window(stft, P):
    spec_windows = None
    
    for col in range(stft.shape[1]):
        left = col - P
        right = col + P + 1
        if left < 0:
            left = np.abs(left)
            padding = np.zeros((stft.shape[0], left))
            win = np.concatenate((padding, stft[:, :right]), axis=1)
        elif right > stft.shape[1]:
            right = right - stft.shape[1]
            padding = np.zeros((stft.shape[0], right))
            win = np.concatenate((stft[:,left:], padding), axis=1)
        else:
            win = stft[:, left:right]
        win = win.flatten('F')   
        
        if col==0:
            spec_windows = win
        else:
            spec_windows = np.column_stack((spec_windows, win))
    return spec_windows


def make_dnn_feats(fpath, noise_path, snr, P, maxlen=1339):
    speech = read(fpath)
    noise = read(noise_path)
    noise = pad_noise(speech, noise)
    blend = generate_noisy(speech, noise, snr)

    mel_clean = librosa.feature.melspectrogram(y=speech, sr=16000,
                                                        n_fft=512,
                                                        hop_length=256,
                                                        n_mels=64)
    mel_noisy = librosa.feature.melspectrogram(y=blend, sr=16000,
                                                        n_fft=512,
                                                        hop_length=256,
                                                        n_mels=64)
    pad_ind = mel_noisy.shape[1]
    feats = pad_inf(window(mel_noisy, P), maxlen)
    print('Feats:', feats)
    target = pad_inf(np.log(mel_clean), maxlen)
    #print("target:", target.shape)

    return {'x': feats, 't':target, 'pad_i':pad_ind}