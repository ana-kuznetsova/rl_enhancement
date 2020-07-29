import numpy as np

#### REWARD DEFINITION ####

def reward():
    pass

def time_weight(Y, S):
    '''
    E - calculated time weight
    Y: predicted spectrogram
    S: true spectrogram
    '''
    E_bar = np.sum(np.abs(np.log(Y) - np.log(S))**2, axis=0)
    return E_bar