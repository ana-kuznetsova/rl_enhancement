import numpy as np
import os
from tqdm import tqdm
import argparse

from models import train_dnn
from models import pretrain


def main(args):
    print('Start DNN mapping...')
    train_dnn(args.num_epochs,
              args.model_path,
              args.x_path,
              args.y_path,
              args.loss_path,
              args.chunk_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs', required=True)
    parser.add_argument('--chunk_size', type=int, help='Number of training examples to load from disk')
    parser.add_argument('--model_path', type=str, help='Dir to save best model', required=True)
    parser.add_argument('--x_path', type=str, help='path to X files', required=True)
    parser.add_argument('--y_path', type=str, help='path to y files', required=True)
    parser.add_argument('--loss_path', type=str, help='Dir to save losses', required=True)
    parser.add_argument('--pretrained', type=str, help='True or false, load weights from pretrained model')


    args = parser.parse_args()
    main(args)