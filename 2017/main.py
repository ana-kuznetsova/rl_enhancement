import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

from models import train_dnn


def main(args):
    print('Start DNN mapping...')
    train_dnn(args.num_epochs,
              args.model_path,
              args.csv_path,
              args.chunk_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_epochs', type=str, help='Number of epochs', required=True)
    parser.add_argument('--chunk_size', type=str, help='Number of training examples to load from disk')
    parser.add_argument('--model_path', type=str, help='Dir to save best model', required=True)
    parser.add_argument('--csv_path', type=str, help='Csv file with X and y mapping', required=True)


    args = parser.parse_args()
    main(args)