import argparse
import sys
import os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import project.data.dataset_utils as data_utils
import project.models.model_utils as model_utils
import project.train.train_utils as train_utils
# import os
import torch
# import datetime
# import cPickle as pickle
# import pdb


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Binary Text Classifier')
    # Learning
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('--loss', type=str, default='MSE', help='type of loss to use')
    # Data loading
    parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
    parser.add_argument('--bow', action='store_true', default=False, help='use bow representation instead of embeddings')
    parser.add_argument('--tfidf', action='store_true', default=False, help='use tfidf weighted bow representation')
    parser.add_argument('--word_embeddings', type=str, help='path to word embeddings')
    parser.add_argument('--data_path', type=str, help='path to dataset (tab separated label, tokenized sentence pairs)')
    parser.add_argument('--max_seq_length', type=int, default=20, help='max length of sentence [default: 20]')
    # Model
    parser.add_argument('--num_hidden', type=int, default=200, help='num hidden units [default: 200]')
    parser.add_argument('--model_name', nargs="?", type=str, default='dan', help="Form of model, i.e dan, rnn, etc.")
    parser.add_argument('--freeze_embeddings', action='store_true', default=False, help='whether or not to calculate gradients and update word embedding weights')
    parser.add_argument('--dropout', type=int, default=0.0, help='If non-zero, introduces a Dropout layer on the outputs of each LSTM / CNN layer except the last layer, with dropout probability equal to dropout. Default: 0')
    parser.add_argument('--kernel_num', type=int, default=128, help='Number of kernels / CNN layers')
    parser.add_argument('--kernel_sizes', nargs='?', type=int, default=[1, 2, 3, 4, 5], help='Sizes of kernels in CNN')
    # Device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('--train', action='store_true', default=False, help='enable train')
    # Task
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--save_path', type=str, default="model.pt", help='Path where to dump model')
    # Parse all terminal arguments
    args = parser.parse_args()
    return args


def main(args):
    # Load data
    train_data, dev_data, embeddings = data_utils.load_dataset(args)
    # Load model
    if args.snapshot is None:
        model = model_utils.get_model(embeddings, args)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            model = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()
    print(model)
    print("\n\nBeginning Training...\n")
    # Train model
    if args.train :
        train_utils.train_model(train_data, dev_data, model, args)
    # Eval model
    else:
        train_utils.eval_model(dev_data, model, args)


if __name__ == '__main__':
    # Update args and print
    args = get_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    main(args)
