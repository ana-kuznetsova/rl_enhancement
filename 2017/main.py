import numpy as np
import librosa
import os
from tqdm import tqdm
import argparse

from models import train_dnn
from models import pretrain
from models import inference
from data import save_imag
from utils import collect_paths
from data import create_noisy_data
from data import calc_masks

def main(args):
    if args.mode=='train':
        print('Start DNN mapping...')
        train_dnn(args.num_epochs,
                args.model_path,
                args.x_path,
                args.y_path,
                args.loss_path,
                args.chunk_size, 
                args.feat_type)
    elif args.mode=='test':
        print('Staring inference on test data...')
        inference(args.test_path, args.test_out, args.model_path, args.imag, args.chunk_size, args.feat_type)
    elif args.mode=='data':

        WIN_LEN = 512
        HOP_SIZE = 256
        FS = 16000
        noise_path = '/N/project/aspire_research_cs/Data/Corpora/Noise/cafe_16k.wav'
        ## Generate stfts of noisy data
        '''
        print('Generating TRAINING data...')
        train_files = collect_paths('/N/project/aspire_research_cs/Data/Corpora/Speech/TIMIT/corpus/')
        out_path = '/N/slate/anakuzne/se_data/snr0_train/'
        noise_path = '/N/project/aspire_research_cs/Data/Corpora/Noise/cafe_16k.wav'

        create_noisy_data(train_files, out_path, noise_path, 0, WIN_LEN, HOP_SIZE, FS)

        print('Generating TEST data...')

        test_files = collect_paths('/N/project/aspire_research_cs/Data/Corpora/Speech/TIMIT/test_corpus/corpus/')
        out_path = '/N/slate/anakuzne/se_data/snr0_test/'

        create_noisy_data(test_files, out_path, noise_path, 0, WIN_LEN, HOP_SIZE, FS)
        '''


        print('Generate TARGET data...')
        target_files = collect_paths('/N/project/aspire_research_cs/Data/Corpora/Speech/TIMIT/corpus/')
        calc_masks(target_files, noise_path, FS, WIN_LEN, HOP_SIZE,
                   mask_dir='/N/slate/anakuzne/se_data/snr0_irm_target/',
                   mask_type='IRM')
        
        #print('Saving phase information')
        #save_imag('/N/slate/anakuzne/se_data/snr0_test/', '/N/slate/anakuzne/se_data/snr0_test_img/')

    elif args.mode=='pretrain':
        pretrain(args.chunk_size, args.model_path, args.x_path, args.y_path, args.loss_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--chunk_size', type=int, help='Number of training examples to load from disk')
    parser.add_argument('--model_path', type=str, help='Dir to save best model')
    parser.add_argument('--x_path', type=str, help='path to X files')
    parser.add_argument('--y_path', type=str, help='path to y files')
    parser.add_argument('--loss_path', type=str, help='Dir to save losses')
    parser.add_argument('--mode', type=str, help='Train or test', required=True)
    parser.add_argument('--test_path', type=str, help='path to test data')
    parser.add_argument('--test_out', type=str, help='Path to dir to save test output')
    parser.add_argument('--clean_test_path', type=str, help='Path to reference test data')
    parser.add_argument('--imag', type=str, help='Path to files with imaginary part')
    parser.add_argument('--feat_type', type=str, help='Features to use')


    args = parser.parse_args()
    main(args)