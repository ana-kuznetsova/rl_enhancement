import pandas as pd 
from pypesq import pesq
from tqdm import tqdm
import os
import soundfile as sf
import librosa
import numpy as np
from data import pad

def eval_pesq(noisy_test, clean_test, out_path, 
    img_path='/N/slate/anakuzne/se_data/snr0_test_img/', 
    fs=16000):

    noisy = os.listdir(noisy_test)
    clean = os.listdir(clean_test)
    imag = os.listdir(img_path)

    scores = []

    print('Calculating PESQ...')
    for f in tqdm(clean[:2]):
        if '.wav' in f:
            reference, sr = librosa.load(clean_test+f, mono=True)
            reference = librosa.core.resample(reference, sr, 16000)
            ind = noisy.index('corpus_'+ f.split('.')[0]+'.npy')
            degraded = np.load(noisy_test+noisy[ind])
            
            ind = imag.index('corpus_'+ f.split('.')[0]+'.npy')
            imag_num = pad(np.load(img_path+noisy[ind]), 1339)
            degraded = degraded + imag_num
            degraded = librosa.istft(degraded, hop_length=256, win_length=512)
            degraded = degraded[:reference.shape[0]]
            #print('degraded:', degraded.shape)
            score = pesq(reference, degraded, fs)
            #print('Test file:', f, 'PESQ: ', score)
            scores.append(score)
    clean = [n for n in clean if '.wav' in n][:2]
    print('len:', len(clean), len(scores))
    data = {'fname':clean, 'PESQ':scores}
    df = pd.DataFrame(data, columns=['fname', 'PESQ'])
    df.to_csv(out_path+'PESQ.csv')
