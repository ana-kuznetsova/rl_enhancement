import pandas as pd 
import pypesq
from tqdm import tqdm
import os
import soundfile as sf

def eval_pesq(noisy_test, clean_test, out_path, fs=44000):
    noisy = os.listdir(noisy_test)
    clean = os.listdir(clean_test)

    scores = []

    print('Calculating PESQ...')
    for f in tqdm(clean_test):
        reference, sr = sf.read(clean_test+f)
        ind = noisy.index(f)
        degraded, sr = sf.read(noisy_test+noisy[ind])
        score = pypesq(reference, degraded, fs)
        print('Test file:', f, 'PESQ: ', score)
        scores.append(score)

    df = pd.DataFrame([clean, scores], columns=['fname', 'PESQ'])
    df.to_csv(out_path+'PESQ.csv')
