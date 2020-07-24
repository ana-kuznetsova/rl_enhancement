import pandas as pd 
import pypesq
from tqdm import tqdm
import os
import soundfile as sf
import librosa
import numpy as np

def eval_pesq(noisy_test, clean_test, out_path, 
    img_path='/N/slate/anakuzne/se_data/snr0_test_img/', 
    fs=16000):

    noisy = os.listdir(noisy_test)
    clean = os.listdir(clean_test)

    scores = []

    print('Calculating PESQ...')
    for f in tqdm(clean):
        if '.wav' in f:
            reference, sr = librosa.load(clean_test+f, mono=True)
            reference = librosa.core.resample(reference, sr, 16000)
            ind = noisy.index('corpus_'+ f.split('.')[0]+'.npy')
            degraded = np.load(noisy_test+noisy[ind])
            imag = np.load(img_path+noisy[ind])
            degraded = degraded + imag
            degraded = librosa.istft(degraded, hop_length=256, win_length=512)
            degraded = degraded[:reference.shape[0]]
            score = pypesq(reference, degraded, fs)
            print('Test file:', f, 'PESQ: ', score)
            scores.append(score)

    df = pd.DataFrame([clean, scores], columns=['fname', 'PESQ'])
    df.to_csv(out_path+'PESQ.csv')
