import numpy as np
import librosa
import os
from tqdm import tqdm
import argparse

from preproc import precalc_Wiener
from preproc import KMeans
from dnn_rl import q_pretrain
from dnn_rl import q_train
from dnn_rl import eval_actions
from models import dnn_predict
from models import pretrain
from models import train_dnn
from qlearn import q_learning


def main(args):
    if args.mode=='data':
        #Precalculate Wiener target
        precalc_Wiener(args.x_path, args.noise_path, args.out_path)
        #Calculate cluster centers
        KMeans(args.x_path, args.out_path, args.k)
    elif args.mode=='dnn_pretrain':
        pretrain(args.x_path, args.model_path, args.num_epochs,
                 args.noise_path, args.snr, args.P, args.resume)
    elif args.mode=='train_dnn':
        train_dnn(args.xpath, args.model_path, args.num_epochs, args.noise_path,
                  args.snr, args.P, args.from_pretrained, args.resume)

    elif args.mode=='qpretrain':
        q_pretrain(args.x_path, args.noise_path, args.cluster_path,
                    args.model_path, args.num_epochs, args.snr, args.P, args.maxlen, args.resume)

    elif args.mode=='qtrain':
        q_train(args.x_path, args.noise_path, args.cluster_path,
                    args.model_path, args.num_epochs, args.snr, args.P, args.maxlen, args.resume)

    elif args.mode=='eval_actions':
        eval_actions(args.x_path, args.noise_path, args.cluster_path, args.model_path, args.snr, args.P)

    elif args.mode=='dnn_pred':
        dnn_predict(args.x_path, args.noise_path, args.model_path, args.out_path, args.snr, args.P)

    elif args.mode=='qlearn':
        q_learning(args.x_path, args.noise_path, args.cluster_path, args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--model_path', type=str, help='Dir to save best model')
    parser.add_argument('--snr', type=int, help="SNR (Db)")
    parser.add_argument('--P', type=int, help='Window size')
    parser.add_argument('--maxlen', type=int, help='Maximum lenth of the input feature (e.g. num frames in spec')
    parser.add_argument('--x_path', type=str, help='path to X files')
    parser.add_argument('--y_path', type=str, help='path to y files')
    parser.add_argument('--out_path', type=str, help='Path to store output of the function')
    parser.add_argument('--mode', type=str, help='mode', required=True)
    parser.add_argument('--noise_path', type=str, help='Dir where the noise wav is stored')
    parser.add_argument('--from_pretrained', type=bool, help='true or false')
    parser.add_argument('--cluster_path', type=str, help='path to the k-means clusters')
    parser.add_argument('--k', type=int, help='Number of K-means clusters')
    parser.add_argument('--resume', type=bool, help='If training should be resumed from the existing checkpoint')
    args = parser.parse_args()
    main(args)