import os
from tqdm import tqdm
import librosa
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from utils import read
from utils import write_npy
from utils import collect_paths
from utils import pad

import torchaudio
from torchaudio import transforms
import torch

def generate_noisy(speech, noise, desired_snr):    
    #calculate energies
    E_speech = np.sum(np.power(speech, 2))
    E_noise = np.sum(np.power(noise, 2))
    
    #calculate b coeff
    b = np.sqrt((E_speech/(np.power(10, (desired_snr/10))))/E_noise)    
    return speech + b*noise


def create_noisy_data(x_path, out_path, noise_path, 
                      win_len=512, hop_size=256, fs=16000):
    noise = read(noise_path, fs)

    target_SNRs = [0, 3, 6]

    fnames = os.listdir(x_path)
    #print('fnames:', fnames)
    for s in target_SNRs:
        for f in tqdm(fnames):
            if '.wav' in f:
                speech = read(x_path+f, fs)
                noise = pad_noise(speech, noise)
                blend = generate_noisy(speech, noise, s)
                mel_spec = librosa.feature.melspectrogram(y=blend, sr=16000,
                                                            n_fft=512,
                                                            hop_length=256,
                                                            n_mels=64)
                #stft = STFT(blend, win_len, hop_size)

                fname = f.split('.')[0]+"_"+str(s)+'.npy'
                np.save(out_path+fname, mel_spec)


def STFT(x, win_len, hop_size, win_type='hann'):
    return librosa.core.stft(x, win_len, hop_size, win_len, win_type)


def mel_spec(x_path, out_path, noisy=False):
    '''
    x_path: path to raw audio
    '''
    noise_path = '/N/project/aspire_research_cs/Data/Corpora/Noise/cafe_16k.wav'
    x_files = os.listdir(x_path)
    for f in tqdm(x_files):
        if ".wav" in f:
            speech = read(x_path+f, 16000)
            if noisy:
                noise = read(noise_path, 16000)
                noise = pad_noise(speech, noise)
                blend = generate_noisy(speech, noise, 0)
                mel_spec = librosa.feature.melspectrogram(y=blend, sr=16000,
                                                        n_fft=512,
                                                        hop_length=256,
                                                        n_mels=64)
            else:
                mel_spec = librosa.feature.melspectrogram(y=speech, sr=16000,
                                                        n_fft=512,
                                                        hop_length=256,
                                                        n_mels=64)

            f = f.split('.')[0] + '.npy'
            np.save(out_path+f, mel_spec)

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

def calc_mel_wiener(x_path, y_path):
    '''
    x_path: path to clean mel specs
    y_path: path to store Wiener filters for mel specs
    '''
    noise = read('/N/project/aspire_research_cs/Data/Corpora/Noise/cafe_16k.wav')
    noise = librosa.feature.melspectrogram(y=noise, sr=16000,
                                                    n_fft=512,
                                                    hop_length=256,
                                                    n_mels=64)
    files = os.listdir(x_path)
    for f in tqdm(files):
        speech = np.load(x_path+f)
        w_mel = Wiener(speech, noise[:,:speech.shape[1]])
        np.save(y_path+f, w_mel)
    

def calc_masks(speech_paths, noise_path, fs, win_len, hop_size,
               mask_dir,
               mask_type='IRM',
               stft_dir=None):
    
    noise = read(noise_path, fs)

    for p in tqdm(speech_paths):
        fname = p.split('/')[-2]+ '_' + p.split('/')[-1].split('.')[0] + '.npy'
        speech = read(p, fs)
        noise = pad_noise(speech, noise)
        stft_noise = STFT(noise, win_len, hop_size)
        stft_clean = STFT(speech, win_len, hop_size)
        if mask_type=='IRM':
            irm = IRM(stft_clean, stft_noise)
            write_npy(mask_dir, fname, irm)
        elif mask_type=='Wiener':
            wiener = Wiener(stft_clean, stft_noise)
            write_npy(mask_dir, fname, wiener)

        elif mask_type=='ln':
            target = np.log(stft_clean) + 0.0000001
            #target = np.nan_to_num(target)
            write_npy(mask_dir, fname, target)
        elif mask_type=='stft':
            write_npy(mask_dir, fname, stft_clean)


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

def make_windows(x_path, a_path, ind, P, win_len, hop_size, fs, names=False):
    chunk_x = os.listdir(x_path)[ind[0]:ind[1]]
    batch_indices = []
    X = 0
    A = 0
    for i, path in enumerate(tqdm(chunk_x)):
        arr = np.load(x_path+path)
        true_a = np.load(a_path+path).reshape(-1,1)
        arr = get_X_batch(arr, P)

        if i==0:
            batch_indices.append(i)
        else:
            batch_indices.append(batch_indices[i-1]+arr.shape[0])
        
        if i ==0:
            X = arr
            A = true_a
            #print(A)
        else:
            X = np.vstack((X, arr))
            A = np.vstack((A, true_a))
    return X, A, batch_indices

def make_batch(x_path, y_path, ind, P, maxlen, win_len, hop_size, feat_type, fs, names=False):
    X = []
    y = []
    chunk_x = os.listdir(x_path)[ind[0]:ind[1]]
    print('Loading training examples...')

    for path in tqdm(chunk_x):
        arr = np.load(x_path+path)
        if feat_type=='stft':
            arr = pad(arr, maxlen)
            arr = np.abs(get_X_batch(arr, P))
        elif feat_type=='mel':
            #arr = mel_spec(arr, win_len, hop_size, fs)
            arr = pad(arr, maxlen)
            arr = np.abs(get_X_batch(arr, P))
        X.extend(arr)

        arr = pad(np.abs(np.load(y_path+path)), maxlen).T
        #predict log of speech
        y.extend(arr)
    X = np.asarray(X)
    y = np.asarray(y)
    if names:
        return X, y, chunk_x      
    return X, y 

def make_batch_test(x_list, ind, P, feat_type, maxlen=1339, win_len=512, hop_size=256, fs=16000):
    X = []
    chunk_x = x_list[ind[0]:ind[1]]
    print('Loading test examples...')
    for path in chunk_x:
        arr = np.load(path)
        if feat_type=='stft':
            arr = pad(arr, maxlen)
            arr = np.abs(get_X_batch(arr, P))
        elif feat_type=='mel':
            arr = mel_spec(arr, win_len, hop_size, fs)
            arr = pad(arr, maxlen)
            arr = np.abs(get_X_batch(arr, P))
        X.extend(arr)
    X = np.asarray(X)
    return X


def save_imag(in_path, out_path):
    names = os.listdir(in_path)
    for f in tqdm(names):
        stft = np.load(in_path+f)
        np.save(out_path+f, stft.imag)



def get_freq_bins(train_paths, ind, maxlen=1339):
    chunk_x = train_paths[ind[0]:ind[1]]
    freqs = 0
    first = True
    for path in tqdm(chunk_x):
        #f = read(path)
        #f = STFT(f, 512, 256)
        
        f = np.load(path).T
        #f = pad(f, maxlen).T
        if first:
            freqs = f
            first = False
        freqs = np.concatenate([freqs, f], axis=0)
            #print('SHape:', freqs.shape)
    return freqs


def KMeans(chunk_size, train_path, out_path):
    kmeans = MiniBatchKMeans(n_clusters=32, 
                         batch_size=128,
                         max_iter=100)

    paths = os.listdir(train_path)
    paths = [train_path+p for p in paths]
    #print('Paths:', paths[:10])
    num_chunk = (4620//chunk_size) + 1
    for chunk in range(num_chunk):
        start = chunk*chunk_size
        end = min(start+chunk_size, 4620)
        print(start, end)
        X = get_freq_bins(paths, [start, end])
        kmeans = kmeans.partial_fit(np.abs(X))

    centers = kmeans.cluster_centers_
    np.save(out_path+'kmeans_centers.npy', centers)

def calc_MMSE_labels(x_path, a_path, clean_path, cluster_path):
    '''
    a_path: dir where ground truth mmse actions will be stored
    '''
    fnames = os.listdir(x_path)
    G_mat = np.load(cluster_path).T

    for f in tqdm(fnames):
        A_t = []
        x_source = np.load(x_path+f)
        x_clean = np.load(clean_path+f)
        
        for timestep in range(x_source.shape[1]):
            sums = []
            for a in range(G_mat.shape[1]):
                diff = np.sum(x_clean[:,timestep] - np.multiply(G_mat[:,a], x_source[:, timestep]))
                sums.append(diff)
            sums = np.asarray(sums)
            A_t.append(np.argmin(sums))
        A_t = np.asarray(A_t)
        np.save(a_path+f, A_t)