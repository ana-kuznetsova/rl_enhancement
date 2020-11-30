import numpy as np

#### REWARD DEFINITION ####

def reward(z_rl, z_map, E):
    '''
    z_rl: predicted G from DNN-RL
    z_map: predicted G from DNN-map
    '''
    R_ = R(z_rl, z_map)
    if R_ > 0:
        return (1 - E)*R_
    else:
        return E*R_


def R(z_rl, z_map, alpha=20):
    return np.tanh(alpha*(z_rl - z_map))

def time_weight(Y, S):
    '''
    E - calculated time weight
    Y: predicted spectrogram
    S: true spectrogram
    '''
    Y = np.nan_to_num(np.abs(Y))
    S = np.nan_to_num(np.abs(S))
    sum_ = np.nan_to_num((Y - S)**2)
    E_approx = np.nan_to_num(np.sum(sum_, axis=0))
    E = E_approx/np.max(E_approx)
    return E