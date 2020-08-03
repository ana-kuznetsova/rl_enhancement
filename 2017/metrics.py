import pandas as pd 
from pypesq import pesq
from tqdm import tqdm
import os
import soundfile as sf
import librosa
import numpy as np
from data import pad
from utils import collect_paths

def eval_pesq(predicted_path, noisy_test, clean_test, out_path, 
    img_path='/u/anakuzne/data/snr0_test_img/', 
    fs=16000):
    '''
    Params:
        predicted: stfts predicted by the model
        noisy test: stfts of noise mixture
        clean test: clean unmixed test signals
    '''

    noisy_ref = os.listdir(noisy_test) #.wav
    clean_ref = collect_paths(clean_test) #.wav
    predicted = os.listdir(predicted_path) #.npy
    imag = os.listdir(img_path) #.npy

    scores_clean_ref = []
    scores_noisy_ref = []

    print('Calculating PESQ...')
    for p in tqdm(clean_ref):
        clean_r, sr = librosa.load(p, mono=True)
        clean_r = librosa.core.resample(clean_r, sr, 16000)
        fname = p.split('/')[-2]+ '_' + p.split('/')[-1].split('.')[0] + '.npy'
        ind = predicted.index(fname)
        pred = np.load(predicted_path+predicted[ind])
            
        ind = imag.index(fname)
        imag_num = pad(np.load(img_path+imag[ind]), 1339)
        pred = pred + imag_num
        pred = librosa.istft(pred, hop_length=256, win_length=512)
        pred = pred[:clean_r.shape[0]]

        ##Compare to clean signal
        score_clean = pesq(clean_r, pred, fs)
        scores_clean_ref.append(score_clean)

        #Compare to degraded signal
        ind = noisy_ref.index(fname)
        noisy_r = np.exp(np.load(noisy_test+noisy_ref[ind]))
        noisy_r = librosa.istft(noisy_r, hop_length=256, win_length=512)
        print('Shapes:', pred.shape, noisy_r.shape)


    '''
    clean = [n for n in clean if '.wav' in n]
    data = {'fname':clean, 'PESQ_clean':scores}
    df = pd.DataFrame(data, columns=['fname', 'PESQ_clean_ref', 'PESQ_noisy_ref'])
    df.to_csv(out_path+'PESQ.csv')
    '''
