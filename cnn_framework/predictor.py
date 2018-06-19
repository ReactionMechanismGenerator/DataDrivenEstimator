#!/usr/bin/env python
# -*- coding:utf-8 -*-

from cnn_framework.cnn_model import build_model, train_model, reset_model, save_model, write_loss_report
from cnn_framework.input import read_input_file
from cnn_framework.molecule_tensor import get_molecule_tensor, pad_molecule_tensor
import os
import shutil
import rmgpy
import numpy as np
from cnn_framework.data import (prepare_data_one_fold, prepare_folded_data_from_multiple_datasets,
                                prepare_full_train_data_from_multiple_datasets, split_inner_val_from_train_data,
                                prepare_folded_data_from_file, prepare_full_train_data_from_file)
import logging
from keras.callbacks import EarlyStopping
import json


class Predictor(object):
    def __init__(self, input_file=None, data_file=None, save_tensors_dir=None, keep_tensors=False, out_dir=None,
                 normalize=False):
        self.model = None
        self.input_file = input_file
        self.data_file = data_file
        self.save_tensors_dir = save_tensors_dir
        self.keep_tensors = keep_tensors
        self.out_dir = out_dir
        self.normalize = normalize
        self.datasets = None
        self.add_extra_atom_attribute = None
        self.add_extra_bond_attribute = None
        self.differentiate_atom_type = None
        self.differentiate_bond_type = None
        self.padding = None
        self.padding_final_size = None
        self.prediction_task = None
        self.y_mean = None
        self.y_std = None

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

    def kfcv_train(self, folds, lr_func, save_model_path, pretrained_weights=None,
                   batch_size=1, nb_epoch=150, patience=10, training_ratio=0.9, testing_ratio=0.0):
        # prepare data for training
        if self.get_data_from_file:
            folded_data = prepare_folded_data_from_file(
                self.data_file, folds,
                add_extra_atom_attribute=self.add_extra_atom_attribute,
                add_extra_bond_attribute=self.add_extra_bond_attribute,
                differentiate_atom_type=self.differentiate_atom_type,
                differentiate_bond_type=self.differentiate_bond_type,
                padding=self.padding,
                padding_final_size=self.padding_final_size,
                save_tensors_dir=self.save_tensors_dir,
                testing_ratio=testing_ratio)
        else:
            folded_data = prepare_folded_data_from_multiple_datasets(
                self.datasets, folds,
                add_extra_atom_attribute=self.add_extra_atom_attribute,
                add_extra_bond_attribute=self.add_extra_bond_attribute,
                differentiate_atom_type=self.differentiate_atom_type,
                differentiate_bond_type=self.differentiate_bond_type,
                padding=self.padding,
                padding_final_size=self.padding_final_size,
                prediction_task=self.prediction_task,
                save_tensors_dir=self.save_tensors_dir)

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
            if self.normalize:
                y_train, y_inner_val, y_outer_val, y_test = self.normalize_output(y_train,
                                                                                  y_inner_val,
                                                                                  y_outer_val,
                                                                                  y_test)
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
                                             patience=patience,
                                             load_from_disk=True if self.save_tensors_dir is not None else False,
                                             save_model_path=save_model_path)

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
            if pretrained_weights is not None:
                self.load_parameters(pretrained_weights)
            else:
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

        # Delete tensor directory
        if self.save_tensors_dir is not None:
            if not self.keep_tensors:
                shutil.rmtree(self.save_tensors_dir)

    def full_train(self, lr_func, save_model_path,
                   batch_size=1, nb_epoch=150, patience=10, training_ratio=0.9, testing_ratio=0.0):
        # prepare data for training
        if self.get_data_from_file:
            split_data = prepare_full_train_data_from_file(
                self.data_file,
                add_extra_atom_attribute=self.add_extra_atom_attribute,
                add_extra_bond_attribute=self.add_extra_bond_attribute,
                differentiate_atom_type=self.differentiate_atom_type,
                differentiate_bond_type=self.differentiate_bond_type,
                padding=self.padding,
                padding_final_size=self.padding_final_size,
                save_tensors_dir=self.save_tensors_dir,
                testing_ratio=testing_ratio,
                meta_dir=self.out_dir
            )
        else:
            split_data = prepare_full_train_data_from_multiple_datasets(
                self.datasets,
                add_extra_atom_attribute=self.add_extra_atom_attribute,
                add_extra_bond_attribute=self.add_extra_bond_attribute,
                differentiate_atom_type=self.differentiate_atom_type,
                differentiate_bond_type=self.differentiate_bond_type,
                padding=self.padding,
                padding_final_size=self.padding_final_size,
                prediction_task=self.prediction_task,
                save_tensors_dir=self.save_tensors_dir,
                meta_dir=self.out_dir
            )

        X_test, y_test, X_train, y_train = split_data

        losses = []
        inner_val_losses = []
        test_losses = []
        data = split_inner_val_from_train_data(X_train, y_train, training_ratio=training_ratio)

        X_train, X_inner_val, y_train, y_inner_val = data

        if self.normalize:
            y_train, y_inner_val, y_test = self.normalize_output(y_train, y_inner_val, y_test)

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
                                         patience=patience,
                                         load_from_disk=True if self.save_tensors_dir is not None else False,
                                         save_model_path=save_model_path)

        model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss = train_model_output

        # loss and inner_val_loss each is a list
        # containing loss for each epoch
        losses.append(loss)
        inner_val_losses.append(inner_val_loss)
        test_losses.append(mean_test_loss)

        # save model and write report
        fpath = os.path.join(save_model_path, 'full_train')
        self.save_model(loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, fpath)

        # Delete tensor directory
        if self.save_tensors_dir is not None:
            if not self.keep_tensors:
                shutil.rmtree(self.save_tensors_dir)

    def kfcv_batch_train(self, folds, pretrained_weights=None,
                         batch_size=50, nb_epoch=150, patience=10, training_ratio=0.9, testing_ratio=0.0):
        # prepare data for training
        if self.get_data_from_file:
            folded_data = prepare_folded_data_from_file(
                self.data_file, folds,
                add_extra_atom_attribute=self.add_extra_atom_attribute,
                add_extra_bond_attribute=self.add_extra_bond_attribute,
                differentiate_atom_type=self.differentiate_atom_type,
                differentiate_bond_type=self.differentiate_bond_type,
                padding=self.padding,
                padding_final_size=self.padding_final_size,
                save_tensors_dir=self.save_tensors_dir,
                testing_ratio=testing_ratio)
        else:
            folded_data = prepare_folded_data_from_multiple_datasets(
                self.datasets, folds,
                add_extra_atom_attribute=self.add_extra_atom_attribute,
                add_extra_bond_attribute=self.add_extra_bond_attribute,
                differentiate_atom_type=self.differentiate_atom_type,
                differentiate_bond_type=self.differentiate_bond_type,
                padding=self.padding,
                padding_final_size=self.padding_final_size,
                prediction_task=self.prediction_task,
                save_tensors_dir=self.save_tensors_dir)

        X_test, y_test, folded_Xs, folded_ys = folded_data

        # Data might be stored as file names
        if len(X_test) > 0:
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

            if isinstance(X_train, np.ndarray):
                X_train = np.concatenate((X_train, X_inner_val))
            else:
                X_train.extend(X_inner_val)
            if isinstance(y_train, np.ndarray):
                y_train = np.concatenate((y_train, y_inner_val))
            else:
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

            if self.normalize:
                y_train, y_outer_val, y_test = self.normalize_output(y_train, y_outer_val, y_test)

            earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')

            history_callback = self.model.fit(np.asarray(X_train),
                                              np.asarray(y_train),
                                              callbacks=[earlyStopping],
                                              nb_epoch=nb_epoch,
                                              batch_size=batch_size,
                                              validation_split=1.0-training_ratio)

            loss_history = history_callback.history
            with open(os.path.join(self.out_dir, 'history.json_fold_{0}'.format(fold)), 'w') as f_in:
                json.dump(loss_history, f_in, indent=2)

            # evaluate outer validation loss
            outer_val_loss = self.model.evaluate(np.asarray(X_outer_val),
                                                 np.asarray(y_outer_val),
                                                 batch_size=50)
            logging.info("\nOuter val loss: {0}".format(outer_val_loss))

            if len(X_test) > 0:
                test_loss = self.model.evaluate(np.asarray(X_test), np.asarray(y_test), batch_size=50)
                logging.info("\nTest loss: {0}".format(test_loss))

            # once finish training one fold, reset the model
            if pretrained_weights is not None:
                self.load_parameters(pretrained_weights)
            else:
                self.reset_model()

        # Delete tensor directory
        if self.save_tensors_dir is not None:
            if not self.keep_tensors:
                shutil.rmtree(self.save_tensors_dir)

    def normalize_output(self, y_train, *other_ys):
        y_train = np.asarray(y_train)
        self.y_mean = np.mean(y_train, axis=0)
        self.y_std = np.std(y_train, axis=0)
        logging.info('Mean: {}, std: {}'.format(self.y_mean, self.y_std))

        y_train = (y_train - self.y_mean) / self.y_std
        other_ys = tuple((np.asarray(y) - self.y_mean) / self.y_std for y in other_ys)
        return (y_train,) + other_ys

    def load_parameters(self, param_path=None, mean_and_std_path=None):
        if param_path is not None:
            self.model.load_weights(param_path)
        if mean_and_std_path is not None:
            npzfile = np.load(mean_and_std_path)
            self.y_mean = npzfile['mean']
            self.y_std = npzfile['std']

    def reset_model(self):
        self.model = reset_model(self.model)

    def save_model(self, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, fpath):
        save_model(self.model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, fpath)
        if self.y_mean is not None and self.y_std is not None:
            np.savez(fpath + '_mean_std.npz', mean=self.y_mean, std=self.y_std)
            logging.info('...saved y mean and standard deviation to {}_mean_std.npz'.format(fpath))

    def predict(self, molecule=None, molecule_tensor=None):
        """
        Predict the output given a molecule. If a tensor is specified, it
        overrides the molecule argument.
        """
        if molecule_tensor is None:
            molecule_tensor = get_molecule_tensor(molecule,
                                                  self.add_extra_atom_attribute,
                                                  self.add_extra_bond_attribute,
                                                  self.differentiate_atom_type,
                                                  self.differentiate_bond_type)
            if self.padding:
                molecule_tensor = pad_molecule_tensor(molecule_tensor, self.padding_final_size)
        molecule_tensor_array = np.array([molecule_tensor])
        y_pred = self.model.predict(molecule_tensor_array)

        if self.y_mean is not None and self.y_std is not None:
            y_pred = y_pred * self.y_std + self.y_mean

        if self.prediction_task == "Cp(cal/mol/K)":
            return y_pred[0]
        else:
            return y_pred[0][0]

    def evaluate(self, X, y):
        """
        Evaluate RMSE and MAE given a list or array of file names or tensors
        and a list or array of outputs.
        """
        y_pred = []
        for x in X:
            if self.save_tensors_dir is not None:
                x = np.load(x)
            y_pred.append(self.predict(molecule_tensor=x))
        y_pred = np.array(y_pred).flatten()
        y = np.asarray(y)
        if self.y_mean is not None and self.y_std is not None:
            y = y * self.y_std + self.y_mean
        y = y.flatten()

        diff = y - y_pred
        rmse = np.sqrt(np.dot(diff.T, diff) / len(y))
        mae = np.sum(np.abs(diff)) / len(y)

        return rmse, mae
