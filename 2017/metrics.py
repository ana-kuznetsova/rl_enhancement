import pandas as pd 
from pypesq import pesq
from tqdm import tqdm
import os
import soundfile as sf
import librosa
import numpy as np
from data import pad
from data import collect_paths

def eval_pesq(noisy_test, clean_test, out_path, 
    img_path='/u/anakuzne/data/snr0_test_img/', 
    fs=16000):

    noisy = collect_paths(noisy_test)
    clean = collect_paths(clean_test)
    imag = collect_paths(img_path)

    scores = []

    print('Calculating PESQ clean refernence...')
    for p in tqdm(clean):
        reference, sr = librosa.load(f, mono=True)
        reference = librosa.core.resample(reference, sr, 16000)
        print('FNAME:', p.split('/')[-2]+ '_' + p.split('/')[-1].split('.')[0] + '.npy')
        ind = noisy.index(p.split('/')[-2]+ '_' + p.split('/')[-1].split('.')[0] + '.npy')
        degraded = np.load(noisy[ind])
            
        ind = imag.index('corpus_'+ f.split('.')[0]+'.npy')
        imag_num = pad(np.load(img_path+noisy[ind]), 1339)
        degraded = degraded + imag_num
        degraded = librosa.istft(degraded, hop_length=256, win_length=512)
        degraded = degraded[:reference.shape[0]]
        #print('degraded:', degraded.shape)
        score = pesq(reference, degraded, fs)
        #print('Test file:', f, 'PESQ: ', score)
        scores.append(score)
    clean = [n for n in clean if '.wav' in n]
    data = {'fname':clean, 'PESQ_clean':scores}
    df = pd.DataFrame(data, columns=['fname', 'PESQ_clean_ref', 'PESQ_noisy_ref'])
    df.to_csv(out_path+'PESQ.csv')
