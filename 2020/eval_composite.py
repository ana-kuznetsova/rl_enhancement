from scipy.io import wavfile
import sys
sys.path.append("../") 
import pysepm
import os
import numpy as np
from tqdm import tqdm


def calc_compositem(clean_path, enhanced_path):
    '''
    Csig: predicted rating of speech distortion:
    Cbak: predicted rating of background distortion;
    Covl: predicted rating of overall quality.
    '''    
    fnames = os.listdir(clean_path)

    csig = []
    cbak = []
    covl = []
    
    for f in fnames:
        fs, clean_speech = wavfile.read(os.path.join(clean_path, f))
        fs, enhanced_speech = wavfile.read(os.path.join(enhanced_path, f))
        res = pysepm.composite(clean_speech, enhanced_speech, fs)
        csig.append(res[0])
        cbak.append(res[1])
        covl.append(res[2])
    
    print("CSIG:", np.mean(np.array(csig)), " CBAK:", np.mean(np.array(cbak)), " COVL:", np.mean(np.array(covl)))


calc_compositem('/Users/tacha/iu_research/speech_enhancement/data/clean_testset_wav/',
                '/Users/tacha/iu_research/speech_enhancement/data/pre_actor_test/')