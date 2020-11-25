import numpy as np
import librosa
import os
from tqdm import tqdm
import argparse

from preproc import precalc_Wiener
from preproc import KMeans

def main(args):
    if args.mode=='data':
        #Precalculate Wiener target
        #precalc_Wiener(args.x_path, args.noise_path, args.out_path)
        #Calculate cluster centers
        KMeans(args.x_path, args.out_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--model_path', type=str, help='Dir to save best model')
    parser.add_argument('--x_path', type=str, help='path to X files')
    parser.add_argument('--y_path', type=str, help='path to y files')
    parser.add_argument('--out_path', type=str, help='Path to store output of the function')
    parser.add_argument('--mode', type=str, help='', required=True)
    parser.add_argument('--nn', type=str, help='type of the model DNN/RL')
    parser.add_argument('--noise_path', type=str, help='Dir where the noise wav is stored')
    parser.add_argument('--from_pretrained', type=str, help='true or false')
    parser.add_argument('--a_path', type=str, help='path to the ground truth actions')
    parser.add_argument('--cluster_path', type=str, help='path to the k-means clusters')
    parser.add_argument('--resume', type=str, help='Bool')
    args = parser.parse_args()
    main(args)