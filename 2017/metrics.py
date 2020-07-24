import pandas as pd 
import pypesq
from tqdm import tqdm
import os
import soundfile as sf
import librosa

def eval_pesq(noisy_test, clean_test, out_path, fs=16000):
    noisy = os.listdir(noisy_test)
    clean = os.listdir(clean_test)

    scores = []

    print('Calculating PESQ...')
    for f in tqdm(clean):
        reference, sr = librosa.load(clean_test+f, mono=True)
        reference = librosa.core.resample(reference, sr, 16000)
        ind = noisy.index(f)
        degraded = librosa.istft(noisy_test+noisy[ind], hop_length=256, win_length=512)
        degraded = degraded[:reference.shape[0]]
        score = pypesq(reference, degraded, fs)
        print('Test file:', f, 'PESQ: ', score)
        scores.append(score)

    df = pd.DataFrame([clean, scores], columns=['fname', 'PESQ'])
    df.to_csv(out_path+'PESQ.csv')
