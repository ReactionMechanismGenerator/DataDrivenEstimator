#!/usr/bin/env python
# -*- coding:utf-8 -*-

from cnn_framework.predictor import Predictor
import os
import sys
import logging
import argparse
import shutil
import time


def parse_command_line_arguments():
    """
    Parse the command-line arguments being passed to RMG Py. This uses the
    :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='FILE',
                        help='a predictor training input file')

    parser.add_argument('-w', '--weights', metavar='H5',
                        help='Saved model weights to continue training on (typically for transfer learning)')

    parser.add_argument('-d', '--data', metavar='FILE',
                        help='A file specifying which datasets to train on. Alternatively, a space-separated .csv file'
                             ' with SMILES and output(s) in the first and subsequent columns, respectively.')

    parser.add_argument('-o', '--out_dir', metavar='DIR', default=os.getcwd(),
                        help='Output directory')

    parser.add_argument('--save_tensors_dir', metavar='DIR',
                        help='Location to save tensors on disk (frees up memory)')

    parser.add_argument('--keep_tensors', action='store_true',
                        help='Do not delete directory containing tensors at end of job')

    parser.add_argument('-f', '--folds', type=int, default=5,
                        help='number of folds for training')

    parser.add_argument('-tr', '--train_ratio', type=float, default=0.9,
                        help='Fraction of training data to use for actual training, rest is early-stopping validation')

    parser.add_argument('-te', '--test_ratio', type=float, default=0.0,
                        help='Fraction of data to use for testing. If loading data from database,'
                             ' test ratios are specified in datasets file')

    parser.add_argument('-t', '--train_mode', default='full_train',
                        help='train mode: currently support in_house and keras for k-fold cross-validation,'
                             ' and full_train for full training')

    parser.add_argument('-bs', '--batch_size', type=int, default=1,
                        help='batch training size')

    parser.add_argument('-lr', '--learning_rate', default='0.0007_30.0',
                        help='two parameters for learning rate')

    parser.add_argument('-ep', '--nb_epoch', type=int, default=150,
                        help='number of epochs for training')

    parser.add_argument('-pc', '--patience', type=int, default=10,
                        help='number of consecutive epochs allowed for loss increase')

    return parser.parse_args()
################################################################################


def initialize_log(verbose, log_file_name):
    """
    Set up a logger to print output to stdout. The
    `verbose` parameter is an integer specifying the amount of log text seen
    at the console; the levels correspond to those of the :data:`logging` module.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(verbose)

    # Create console handler and set level to debug; send everything to stdout
    # rather than stderr
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(verbose)

    logging.addLevelName(logging.CRITICAL, 'Critical: ')
    logging.addLevelName(logging.ERROR, 'Error: ')
    logging.addLevelName(logging.WARNING, 'Warning: ')
    logging.addLevelName(logging.INFO, '')
    logging.addLevelName(logging.DEBUG, '')
    logging.addLevelName(1, '')

    # Create formatter and add to console handler
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    # formatter = Formatter('%(message)s', '%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(levelname)s%(message)s')
    ch.setFormatter(formatter)

    # create file handler
    if os.path.exists(log_file_name):
        backup = os.path.join(log_file_name[:-9], 'train_backup.log')
        if os.path.exists(backup):
            print("Removing old "+backup)
            os.remove(backup)
        print('Moving {0} to {1}\n'.format(log_file_name, backup))
        shutil.move(log_file_name, backup)
    fh = logging.FileHandler(filename=log_file_name)  #, backupCount=3)
    fh.setLevel(min(logging.DEBUG,verbose))  # always at least VERBOSE in the file
    fh.setFormatter(formatter)
    # notice that STDERR does not get saved to the log file
    # so errors from underlying libraries (eg. openbabel) etc. that report
    # on stderr will not be logged to disk.

    # remove old handlers!
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Add console and file handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

################################################################################


if __name__ == '__main__':

    # to run the script
    # example command:
    # python train_cnn.py -i input.py -d datasets.txt -f 5 -t in_house -bs 1 -lr 0.0007_30.0

    args = parse_command_line_arguments()
    input_file = args.input
    weights_file = args.weights
    data_file = args.data
    out_dir = args.out_dir
    save_tensors_dir = args.save_tensors_dir
    keep_tensors = args.keep_tensors
    folds = args.folds
    training_ratio = args.train_ratio
    testing_ratio = args.test_ratio
    train_mode = args.train_mode
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    patience = args.patience
    lr0, lr1 = [float(i) for i in args.learning_rate.split('_')]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    level = logging.INFO
    initialize_log(level, os.path.join(out_dir, 'train.log'))

    # Log start timestamp
    logging.info('CNN training initiated at ' + time.asctime() + '\n')

    from rmgpy.rmg.main import RMG
    rmg = RMG()
    rmg.logHeader()

    predictor = Predictor(data_file=data_file,
                          save_tensors_dir=save_tensors_dir,
                          keep_tensors=keep_tensors,
                          out_dir=out_dir)
    predictor.load_input(input_file)
    if weights_file is not None:
        predictor.load_parameters(weights_file)

    lr_func = "float({0} * np.exp(- epoch / {1}))".format(lr0, lr1)
    save_model_path = os.path.join(out_dir, 'saved_model')
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    if train_mode == 'in_house':
        predictor.kfcv_train(folds=folds,
                             batch_size=batch_size,
                             lr_func=lr_func,
                             save_model_path=save_model_path,
                             nb_epoch=nb_epoch,
                             patience=patience,
                             training_ratio=training_ratio,
                             testing_ratio=testing_ratio,
                             pretrained_weights=weights_file)
    elif train_mode == 'keras':
        predictor.kfcv_batch_train(folds=folds, batch_size=batch_size,
                                   nb_epoch=nb_epoch,
                                   patience=patience,
                                   training_ratio=training_ratio,
                                   testing_ratio=testing_ratio,
                                   pretrained_weights=weights_file)
    elif train_mode == 'full_train':
        predictor.full_train(batch_size=batch_size,
                             lr_func=lr_func,
                             save_model_path=save_model_path,
                             nb_epoch=nb_epoch,
                             patience=patience,
                             training_ratio=training_ratio,
                             testing_ratio=testing_ratio)
    else:
        raise Exception('Currently not supporting train mode: {0}'.format(train_mode))
