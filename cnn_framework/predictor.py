#!/usr/bin/env python
# -*- coding:utf-8 -*-

from cnn_framework.cnn_model import build_model, train_model, reset_model, save_model, write_loss_report
from cnn_framework.input import read_input_file
from cnn_framework.molecule_tensor import get_molecule_tensor, pad_molecule_tensor
import os
import rmgpy
import numpy as np
from cnn_framework.data import (prepare_data_one_fold, prepare_folded_data_from_multiple_datasets,
                                prepare_full_train_data_from_multiple_datasets, split_inner_val_from_train_data,
                                prepare_folded_data_from_file, prepare_full_train_data_from_file)
import logging
from keras.callbacks import EarlyStopping
import json


class Predictor(object):
    def __init__(self, input_file=None, data_file=None, save_tensors_dir=None):
        self.model = None
        self.input_file = input_file
        self.data_file = data_file
        self.save_tensors_dir = save_tensors_dir
        self.datasets = None
        self.add_extra_atom_attribute = None
        self.add_extra_bond_attribute = None
        self.differentiate_atom_type = None
        self.differentiate_bond_type = None
        self.padding = None
        self.padding_final_size = None
        self.prediction_task = None

        if self.input_file is not None:
            read_input_file(self.input_file, self)

        self.get_data_from_file = False
        if self.data_file is not None:
            if self.data_file.endswith('.csv'):
                self.get_data_from_file = True
            else:
                self.specify_datasets(self.data_file)

    def build_model(self):
        """
        This method is intended to provide a way to build default model
        """
        self.model = build_model()

    def load_input(self, input_file):
        """
        This method is intended to provide a way to build model from an input file
        """
        if input_file:
            self.input_file = input_file

        read_input_file(self.input_file, self)

    def specify_datasets(self, datasets_file_path=None):
        """
        This method specifies which datasets to use for training
        """
        self.datasets = []
        with open(datasets_file_path, 'r') as f_in:
            for line in f_in:
                line = line.strip()
                if line and not line.startswith('#'):
                    dataset, testing_ratio = [token.strip() for token in line.split(':')]
                    host, db, table = [token.strip() for token in dataset.split('.')]
                    self.datasets.append((host, db, table, float(testing_ratio)))

    def kfcv_train(self, folds, lr_func, save_model_path,
                   batch_size=1, nb_epoch=150, patience=10, training_ratio=0.9, testing_ratio=0.0):
        # prepare data for training
        if self.get_data_from_file:
            folded_data = prepare_folded_data_from_file(self.data_file, folds,
                                                        self.add_extra_atom_attribute,
                                                        self.add_extra_bond_attribute,
                                                        self.differentiate_atom_type,
                                                        self.differentiate_bond_type,
                                                        self.padding,
                                                        self.padding_final_size,
                                                        self.save_tensors_dir,
                                                        testing_ratio)
        else:
            folded_data = prepare_folded_data_from_multiple_datasets(self.datasets, folds,
                                                                     self.add_extra_atom_attribute,
                                                                     self.add_extra_bond_attribute,
                                                                     self.differentiate_atom_type,
                                                                     self.differentiate_bond_type,
                                                                     self.padding,
                                                                     self.padding_final_size,
                                                                     self.prediction_task,
                                                                     self.save_tensors_dir)

        X_test, y_test, folded_Xs, folded_ys = folded_data

        losses = []
        inner_val_losses = []
        outer_val_losses = []
        test_losses = []
        for fold in range(folds):
            data = prepare_data_one_fold(folded_Xs,
                                         folded_ys,
                                         current_fold=fold,
                                         shuffle_seed=4,
                                         training_ratio=training_ratio)

            # execute train_model
            X_train, X_inner_val, X_outer_val, y_train, y_inner_val, y_outer_val = data
            train_model_output = train_model(self.model,
                                             X_train,
                                             y_train,
                                             X_inner_val,
                                             y_inner_val,
                                             X_test,
                                             y_test,
                                             X_outer_val,
                                             y_outer_val,
                                             nb_epoch=nb_epoch,
                                             batch_size=batch_size,
                                             lr_func=lr_func,
                                             patience=patience)

            model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss = train_model_output

            # loss and inner_val_loss each is a list
            # containing loss for each epoch
            losses.append(loss)
            inner_val_losses.append(inner_val_loss)
            outer_val_losses.append(mean_outer_val_loss)
            test_losses.append(mean_test_loss)

            # save model and write fold report
            fpath = os.path.join(save_model_path, 'fold_{0}'.format(fold))
            self.save_model(loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, fpath)

            # once finish training one fold, reset the model
            self.reset_model()

        # mean inner_val_loss and outer_val_loss used for selecting parameters,
        # e.g., lr, epoch, attributes, etc
        full_folds_mean_loss = np.mean([l[-1] for l in losses if len(l) > 0])
        full_folds_mean_inner_val_loss = np.mean([l[-1] for l in inner_val_losses if len(l) > 0])
        full_folds_mean_outer_val_loss = np.mean(outer_val_losses)
        full_folds_mean_test_loss = np.mean(test_losses)

        full_folds_loss_report_path = os.path.join(save_model_path, 'full_folds_loss_report.txt')

        write_loss_report(full_folds_mean_loss,
                          full_folds_mean_inner_val_loss,
                          full_folds_mean_outer_val_loss,
                          full_folds_mean_test_loss,
                          full_folds_loss_report_path)

    def full_train(self, lr_func, save_model_path,
                   batch_size=1, nb_epoch=150, patience=10, training_ratio=0.9, testing_ratio=0.0):
        # prepare data for training
        if self.get_data_from_file:
            split_data = prepare_full_train_data_from_file(self.data_file,
                                                           self.add_extra_atom_attribute,
                                                           self.add_extra_bond_attribute,
                                                           self.differentiate_atom_type,
                                                           self.differentiate_bond_type,
                                                           self.padding,
                                                           self.padding_final_size,
                                                           self.save_tensors_dir,
                                                           testing_ratio)
        else:
            split_data = prepare_full_train_data_from_multiple_datasets(self.datasets,
                                                                        self.add_extra_atom_attribute,
                                                                        self.add_extra_bond_attribute,
                                                                        self.differentiate_atom_type,
                                                                        self.differentiate_bond_type,
                                                                        self.padding,
                                                                        self.padding_final_size,
                                                                        self.prediction_task,
                                                                        self.save_tensors_dir)

        X_test, y_test, X_train, y_train = split_data

        losses = []
        inner_val_losses = []
        test_losses = []
        data = split_inner_val_from_train_data(X_train, y_train, training_ratio=training_ratio)

        X_train, X_inner_val, y_train, y_inner_val = data

        # execute train_model
        logging.info('\nStart full training...')
        logging.info('Training data: {} points'.format(len(X_train)))
        logging.info('Inner val data: {} points'.format(len(X_inner_val)))
        logging.info('Test data: {} points'.format(len(X_test)))
        train_model_output = train_model(self.model,
                                         X_train,
                                         y_train,
                                         X_inner_val,
                                         y_inner_val,
                                         X_test,
                                         y_test,
                                         X_outer_val=None,
                                         y_outer_val=None,
                                         nb_epoch=nb_epoch,
                                         batch_size=batch_size,
                                         lr_func=lr_func,
                                         patience=patience)

        model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss = train_model_output

        # loss and inner_val_loss each is a list
        # containing loss for each epoch
        losses.append(loss)
        inner_val_losses.append(inner_val_loss)
        test_losses.append(mean_test_loss)

        # save model and write report
        fpath = os.path.join(save_model_path, 'full_train')
        self.save_model(loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, fpath)

    def kfcv_batch_train(self, folds, batch_size=50, nb_epoch=150, patience=10, training_ratio=0.9, testing_ratio=0.0):
        # prepare data for training
        if self.get_data_from_file:
            folded_data = prepare_folded_data_from_file(self.data_file, folds,
                                                        self.add_extra_atom_attribute,
                                                        self.add_extra_bond_attribute,
                                                        self.differentiate_atom_type,
                                                        self.differentiate_bond_type,
                                                        self.padding,
                                                        self.padding_final_size,
                                                        self.save_tensors_dir,
                                                        testing_ratio)
        else:
            folded_data = prepare_folded_data_from_multiple_datasets(self.datasets, folds,
                                                                     self.add_extra_atom_attribute,
                                                                     self.add_extra_bond_attribute,
                                                                     self.differentiate_atom_type,
                                                                     self.differentiate_bond_type,
                                                                     self.padding,
                                                                     self.padding_final_size,
                                                                     self.prediction_task,
                                                                     self.save_tensors_dir)

        X_test, y_test, folded_Xs, folded_ys = folded_data

        # Data might be stored as file names
        if isinstance(X_test[0], str):
            dims = np.load(X_test[0]).shape
            X_test_new = np.zeros((len(X_test),) + dims)
            for i, fname in enumerate(X_test):
                X_test_new[i] = np.load(fname)
            X_test = X_test_new

        for fold in range(folds):
            data = prepare_data_one_fold(folded_Xs,
                                         folded_ys,
                                         current_fold=fold,
                                         shuffle_seed=4,
                                         training_ratio=training_ratio)

            X_train, X_inner_val, X_outer_val, y_train, y_inner_val, y_outer_val = data

            X_train.extend(X_inner_val)
            y_train.extend(y_inner_val)

            # Data might be stored as file names
            if isinstance(X_train[0], str):
                dims = np.load(X_train[0]).shape
                X_train_new = np.zeros((len(X_train),) + dims)
                X_outer_val_new = np.zeros((len(X_outer_val),) + dims)
                for i, fname in enumerate(X_train):
                    X_train_new[i] = np.load(fname)
                for i, fname in enumerate(X_outer_val):
                    X_outer_val_new[i] = np.load(fname)
                X_train = X_train_new
                X_outer_val = X_outer_val_new

            earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')

            history_callback = self.model.fit(np.asarray(X_train),
                                              np.asarray(y_train),
                                              callbacks=[earlyStopping],
                                              nb_epoch=nb_epoch,
                                              batch_size=batch_size,
                                              validation_split=0.1)

            loss_history = history_callback.history
            with open('history.json_fold_{0}'.format(fold), 'w') as f_in:
                json.dump(loss_history, f_in, indent=2)

            # evaluate outer validation loss
            outer_val_loss = self.model.evaluate(np.asarray(X_outer_val),
                                                 np.asarray(y_outer_val),
                                                 batch_size=50)
            logging.info("\nOuter val loss: {0}".format(outer_val_loss))

            test_loss = self.model.evaluate(np.asarray(X_test), np.asarray(y_test), batch_size=50)
            logging.info("\nTest loss: {0}".format(test_loss))

            # once finish training one fold, reset the model
            self.reset_model()

    def load_parameters(self, param_path=None):
        if not param_path:
            param_path = os.path.join(os.path.dirname(rmgpy.__file__),
                                      'cnn_framework',
                                      'data',
                                      'weights',
                                      'polycyclic_enthalpy_weights.h5'
                                      )

        self.model.load_weights(param_path)

    def reset_model(self):
        self.model = reset_model(self.model)

    def save_model(self, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, fpath):
        save_model(self.model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, fpath)

    def predict(self, molecule):
        molecule_tensor = get_molecule_tensor(molecule,
                                              self.add_extra_atom_attribute,
                                              self.add_extra_bond_attribute,
                                              self.differentiate_atom_type,
                                              self.differentiate_bond_type)
        if self.padding:
            molecule_tensor = pad_molecule_tensor(molecule_tensor, self.padding_final_size)
        molecule_tensor_array = np.array([molecule_tensor])
        if self.prediction_task == "Cp(cal/mol/K)":
            return self.model.predict(molecule_tensor_array)[0]
        else:
            return self.model.predict(molecule_tensor_array)[0][0]
