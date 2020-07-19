import numpy as np
import os
from tqdm import tqdm
import argparse

from models import train_dnn
from models import pretrain
from models import inference
from data import save_imag

def main(args):
    if args.mode=='train':
        print('Start DNN mapping...')
        train_dnn(args.num_epochs,
                args.model_path,
                args.x_path,
                args.y_path,
                args.loss_path,
                args.chunk_size)
    elif args.mode=='test':
        print('Staring inference on test data...')
        inference(args.test_path, args.clean_test_path, args.test_out, args.model_path, args.imag, args.chunk_size)
    elif args.mode=='data':
        print('Saving phase information')
        save_imag(args.test_path, args.test_out)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--chunk_size', type=int, help='Number of training examples to load from disk')
    parser.add_argument('--model_path', type=str, help='Dir to save best model')
    parser.add_argument('--x_path', type=str, help='path to X files')
    parser.add_argument('--y_path', type=str, help='path to y files')
    parser.add_argument('--loss_path', type=str, help='Dir to save losses')
    parser.add_argument('--pretrained', type=str, help='True or false, load weights from pretrained model')
    parser.add_argument('--mode', type=str, help='Train or test', required=True)
    parser.add_argument('--test_path', type=str, help='path to test data')
    parser.add_argument('--test_out', type=str, help='Path to dir to save test output')
    parser.add_argument('--clean_test_path', type=str, help='Path to reference test data')
    parser.add_argument('--imag', type=str, help='Path to files with imaginary part')
    parser.add_argument('--feat_type', type=str, help='Features to use')


    args = parser.parse_args()
    main(args)