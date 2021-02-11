import torchaudio
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
            noise = torch.cat((noise, noise[:diff]), 0)
        else:
            noise = torch.tile(noise, repeat)
            diff = speech_len - noise.shape[0]
            noise = torch.cat((noise, noise[:diff]), 0)           
            
    elif speech_len < noise_len:
        noise = noise[:speech_len]  
    return noise

def generate_noisy(speech, noise, desired_snr):    
    #calculate energies
    E_speech = torch.sum(speech**2)
    E_noise = torch.sum(noise**2)
    
    #calculate b coeff
    b = torch.sqrt((E_speech/(10**(desired_snr/10)))/E_noise)    
    return speech + b*noise

def get_features(source, noise, snr):
    source = torchaudio.load_wav(source)
    noise = torchaudio.load_wav(noise)
    noise = pad_noise(source, noise)

    MFCC = torchaudio.transforms.MFCC(n_mfcc=64)


    mfcc_source = MFCC(source)
    mfcc_noise = MFCC(noise)

    blend = generate_noisy(mfcc_source, mfcc_noise, snr)
