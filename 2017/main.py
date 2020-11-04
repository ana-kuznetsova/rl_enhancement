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
from data import KMeans
from data import calc_MMSE_labels
from data import mel_spec
from data import calc_mel_wiener
from metrics import eval_pesq
from qlearn import q_learning
from dnn_rl import MMSE_pretrain

def main(args):
    if args.mode=='train':
        print('Start DNN mapping...')
        train_dnn(args.chunk_size, 
                args.model_path,
                args.x_path,
                args.y_path,
                args.pre_path,
                args.from_pretrained)
    elif args.mode=='test':
        print('Staring inference on test data...')
        inference(args.chunk_size, args.x_path, args.y_path, 
                 args.model_path, args.test_out)
    elif args.mode=='data':

        WIN_LEN = 512
        HOP_SIZE = 256
        FS = 16000
        noise_path = '/N/project/aspire_research_cs/Data/Corpora/Noise/cafe_16k.wav'
        
        create_noisy_data('/N/project/aspire_research_cs/Data/Corpora/Speech/TIMIT/test_corpus/corpus/',
                          '/N/slate/anakuzne/se_data/snr0_test_melspecs', noise_path)

        #calc_mel_wiener(args.x_path, args.y_path)

        '''
        calc_MMSE_labels(
            x_path='/nobackup/anakuzne/data/snr0_train_melspecs/',
            a_path='/nobackup/anakuzne/data/action_labels/',
            clean_path = '/nobackup/anakuzne/data/clean_melspecs/',
            cluster_path = '/nobackup/anakuzne/data/kmeans_centers.npy'
        )

        
        print('Generating TRAINING data...')
        train_files = collect_paths('/u/anakuzne/data/TIMIT_full/train/')
        out_path = '/u/anakuzne/data/snr0_train/'

        create_noisy_data(train_files, out_path, noise_path, 0, WIN_LEN, HOP_SIZE, FS)

        print('Generating TEST data...')

        test_files = collect_paths('/u/anakuzne/data/TIMIT_full/test/')
        out_path = '/u/anakuzne/data/snr0_test/'

        create_noisy_data(test_files, out_path, noise_path, 0, WIN_LEN, HOP_SIZE, FS)

       
        print('Generate TARGET data...')
        
        target_files = '/N/project/aspire_research_cs/Data/Corpora/Speech/TIMIT/corpus/'
        calc_masks(target_files, noise_path, FS, WIN_LEN, HOP_SIZE,
                   mask_dir='/N/slate/anakuzne/se_data/snr0_train_clean/',
                   mask_type='ln')
        
        print('Saving phase information')
        save_imag('/N/slate/anakuzne/se_data/snr0_train/', '/N/slate/anakuzne/se_data/snr0_train_img/')
        '''

    elif args.mode=='pretrain':
        pretrain(args.chunk_size, args.model_path, args.x_path, args.y_path, resume=args.resume)

    elif args.mode=='eval':
        eval_pesq(args.preds_path, args.y_path, args.test_path, args.test_out, args.imag)
        '''
        preds_path: stfts predicted by the model
        y_path: stfts of noise mixture
        test_path: clean unmixed test signals
        test_out: where to output file with the results
        '''
    elif args.mode=='cluster':
        KMeans(args.chunk_size, args.x_path, args.test_out)

    elif args.mode=='qlearn':
        q_learning(num_episodes=10,
                   x_path='/nobackup/anakuzne/data/snr036_mel_train/',
                   cluster_path='/nobackup/anakuzne/data/kmeans_centers.npy',
                   model_path= '/nobackup/anakuzne/data/qlearning/',
                   clean_path='/nobackup/anakuzne/data/snr0_train_clean/')
    elif args.mode=='mse_pretrain':
           MMSE_pretrain(chunk_size=500, x_path='/nobackup/anakuzne/data/TIMIT/corpus/',
            a_path='/nobackup/anakuzne/data/action_labels/',
            model_path='/nobackup/anakuzne/data/window_test_q/')




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
    parser.add_argument('--from_pretrained', type=str, help='true or false')
    parser.add_argument('--preds_path', type=str, help='Path to the predicted output')
    parser.add_argument('--pre_path', type=str, help='paths to pretrained model')
    parser.add_argument('--mask', type=str, help='mask type')
    parser.add_argument('--clean_path', type=str, help='path to clean')
    parser.add_argument('--a_path', type=str, help='path to the ground truth actions')
    parser.add_argument('--cluster_path', type=str, help='path to the k-means clusters')
    parser.add_argument('--resume', type=str, help='Bool')
    args = parser.parse_args()
    main(args)