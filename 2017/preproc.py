import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from utils import read
from utils import pad
import torch



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
    b = np.sqrt((E_speech/(np.power(10, desired_snr/10)))/E_noise)    
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
    
    feats = pad(window(mel_noisy, P), maxlen).T
    target = np.log(mel_clean)
    mask = pad(np.ones((target.shape[0], target.shape[1])), maxlen).T
    target = pad(target, maxlen).T
    return {'x': torch.tensor(feats), 't':torch.tensor(target), 'mask':torch.tensor(mask)}


def invert_mel(mel_spec):
    inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=16000, n_fft=512)
    return inv


def Wiener(speech, noise):
    t = (np.abs(speech)**2)/(np.abs(speech)**2+np.abs(noise)**2)
    return t

def precalc_Wiener(x_path, noise_path, out_path):
    fnames = os.listdir(x_path)
    noise = read(noise_path)

    for f in tqdm(fnames):
        speech = read(os.path.join(x_path, f))
        noise = pad_noise(speech, noise)
        mel_clean = librosa.feature.melspectrogram(y=speech, sr=16000,
                                                        n_fft=512,
                                                        hop_length=256,
                                                        n_mels=64)

        mel_noise = librosa.feature.melspectrogram(y=noise, sr=16000,
                                                        n_fft=512,
                                                        hop_length=256,
                                                        n_mels=64)

        w_filter = Wiener(mel_clean, mel_noise)
        f = f.split('.')[0]+'.npy'
        np.save(os.path.join(out_path, f), w_filter)


def KMeans(target_path, out_path):
    '''
    Args:
        target_path: directory with precalculated Wiener filters
        out_path: directory to save cluster centers
    '''
    kmeans = MiniBatchKMeans(n_clusters=32, 
                             batch_size=128,
                             max_iter=100)

    fnames = os.listdir(target_path)
    
    for f in tqdm(fnames):
        w = np.load(os.path.join(target_path, f)).T
        kmeans = kmeans.partial_fit(w)

    centers = kmeans.cluster_centers_
    np.save(os.path.join(out_path, 'kmeans_centers.npy'), centers)


def q_transform(fname, noise_path, cluster_path, snr, P, maxlen=1339):
    G_mat = np.load(cluster_path).T
    A_t = []

    speech = read(fname)
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
    for timestep in range(mel_noisy.shape[1]):
        sums = []
        for a in range(G_mat.shape[1]):
            diff = np.sum(mel_clean[:,timestep] - np.multiply(G_mat[:,a], mel_noisy[:, timestep]))
            sums.append(diff)
        sums = np.asarray(sums)
        A_t.append(np.argmin(sums))
    A_t = np.asarray(A_t)

    print("A_t:", A_t.shape)
    
    feats = pad(window(mel_noisy, P), maxlen).T
    print("feats:", feats.shape)
    target = np.pad(A_t, ((0, 0), (0, maxlen-A_t.shape[1])), 
                    mode='constant', constant_values=(-1, -1))
    return {"x":feats, "t":target}