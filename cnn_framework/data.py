#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import os
import numpy as np
from cnn_framework.molecule_tensor import get_molecule_tensor
from pymongo import MongoClient
from nose.plugins.attrib import attr
from rmgpy.molecule import Molecule


def get_host_info(host):

    host_list = {"rmg":
                     {"host_connection_url": "mongodb://user:user@rmg.mit.edu/admin",
                      "port": 27018},
                 "erebor":
                     {"host_connection_url": "mongodb://user:user@erebor.mit.edu/admin",
                      "port": 27017}}

    return host_list[host]['host_connection_url'], host_list[host]['port']


def get_db_mols(host, db_name, collection_name):

    # connect to db and query
    host_connection_url, port = get_host_info(host)
    client = MongoClient(host_connection_url, port)
    db = getattr(client, db_name)
    collection = getattr(db, collection_name)
    db_cursor = collection.find()

    # collect data
    logging.info('Collecting data: {0}.{1}...'.format(db_name, collection_name))
    db_mols = []
    for db_mol in db_cursor:
        db_mols.append(db_mol)

    logging.info('Done collecting data: {0} points...'.format(len(db_mols)))

    return db_mols


def get_data_from_db(host, db_name, collection_name, prediction_task="Hf298(kcal/mol)"):
    # connect to db and query
    host_connection_url, port = get_host_info(host)
    client = MongoClient(host_connection_url, port)
    db = getattr(client, db_name)
    collection = getattr(db, collection_name)
    db_cursor = collection.find()

    # collect data
    logging.info('Collecting data: {0}.{1}...'.format(db_name, collection_name))
    mols = []
    y = []
    smis = []
    db_mols = []
    for db_mol in db_cursor:
        db_mols.append(db_mol)

    logging.info('Done collecting data: {0} points...'.format(len(db_mols)))

    # decide what predict task is
    if prediction_task not in ["Hf298(kcal/mol)", "S298(cal/mol/K)", "Cp(cal/mol/K)"]:
        raise NotImplementedError("Prediction task: {0} not supported yet!".format(prediction_task))

    for db_mol in db_mols:
        smile = str(db_mol["SMILES_input"])
        if 'adjacency_list' in db_mol:
            adj = str(db_mol["adjacency_list"])
            mol = Molecule().fromAdjacencyList(adj)
        else:
            mol = Molecule().fromSMILES(smile)

        if prediction_task != "Cp(cal/mol/K)":
            yi = float(db_mol[prediction_task])
        else:
            Cp300 = float(db_mol["Cp300(cal/mol/K)"])
            Cp400 = float(db_mol["Cp400(cal/mol/K)"])
            Cp500 = float(db_mol["Cp500(cal/mol/K)"])
            Cp600 = float(db_mol["Cp600(cal/mol/K)"])
            Cp800 = float(db_mol["Cp800(cal/mol/K)"])
            Cp1000 = float(db_mol["Cp1000(cal/mol/K)"])
            Cp1500 = float(db_mol["Cp1500(cal/mol/K)"])
            yi = np.array([Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500])

        mols.append(mol)
        y.append(yi)
        smis.append(smile)

    logging.info('Done: generated {0} molecules'.format(len(mols)))

    return mols, y, smis


def prepare_folded_data_from_multiple_datasets(datasets,
                                               folds,
                                               add_extra_atom_attribute=True,
                                               add_extra_bond_attribute=True,
                                               differentiate_atom_type=True,
                                               differentiate_bond_type=True,
                                               padding=True,
                                               padding_final_size=20,
                                               prediction_task="Hf298(kcal/mol)",
                                               save_tensors_dir=None):
    if save_tensors_dir is not None:
        if not os.path.exists(save_tensors_dir):
            os.makedirs(save_tensors_dir)

    folded_datasets = []
    test_data_datasets = []
    fidx = 0
    for host, db, table, testing_ratio in datasets:
        X, y, _ = get_data_from_db(host, db, table, prediction_task=prediction_task)

        # At this point, X just contains molecule objects,
        # so convert them to tensors. If save_tensors_dir is specified,
        # only store the file names in X and save the tensors to the disk.
        if save_tensors_dir is None:
            X = [get_molecule_tensor(mol,
                                     add_extra_atom_attribute=add_extra_atom_attribute,
                                     add_extra_bond_attribute=add_extra_bond_attribute,
                                     differentiate_atom_type=differentiate_atom_type,
                                     differentiate_bond_type=differentiate_bond_type,
                                     padding=padding,
                                     padding_final_size=padding_final_size)
                 for mol in X]
        else:
            X_new = []
            for mol in X:
                x = get_molecule_tensor(mol,
                                        add_extra_atom_attribute=add_extra_atom_attribute,
                                        add_extra_bond_attribute=add_extra_bond_attribute,
                                        differentiate_atom_type=differentiate_atom_type,
                                        differentiate_bond_type=differentiate_bond_type,
                                        padding=padding,
                                        padding_final_size=padding_final_size)
                fname = os.path.abspath(os.path.join(save_tensors_dir, '{}.npy'.format(fidx)))
                np.save(fname, x)
                X_new.append(fname)
                fidx += 1
            X = X_new

        logging.info('Splitting dataset with testing ratio of {0}...'.format(testing_ratio))
        split_data = split_test_from_train_and_val(X,
                                                   y,
                                                   shuffle_seed=0,
                                                   testing_ratio=testing_ratio)

        (X_test, y_test, X_train_and_val, y_train_and_val) = split_data

        test_data_datasets.append((X_test, y_test))
        (folded_Xs, folded_ys) = prepare_folded_data(X_train_and_val, y_train_and_val, folds, shuffle_seed=2)
        folded_datasets.append((folded_Xs, folded_ys))

    # merge into one folded_Xs and folded_ys
    logging.info('Merging {} datasets for training...'.format(len(datasets)))
    (folded_Xs, folded_ys) = folded_datasets[0]
    if len(folded_datasets) > 1:
        for folded_Xs_1, folded_ys_1 in folded_datasets[1:]:
            folded_Xs_ext = []
            folded_ys_ext = []
            for idx, folded_X in enumerate(folded_Xs):
                folded_X.extend(folded_Xs_1[idx])
                folded_Xs_ext.append(folded_X)

                folded_y = folded_ys[idx]
                folded_y.extend(folded_ys_1[idx])
                folded_ys_ext.append(folded_y)

            folded_Xs = folded_Xs_ext
            folded_ys = folded_ys_ext

    # merge into one X_test and y_test
    (X_test, y_test) = test_data_datasets[0]
    if len(test_data_datasets) > 1:
        for X_test_1, y_test_1 in test_data_datasets[1:]:
            X_test.extend(X_test_1)
            y_test.extend(y_test_1)

    return X_test, y_test, folded_Xs, folded_ys


def prepare_full_train_data_from_multiple_datasets(datasets,
                                                   add_extra_atom_attribute=True,
                                                   add_extra_bond_attribute=True,
                                                   differentiate_atom_type=True,
                                                   differentiate_bond_type=True,
                                                   padding=True,
                                                   padding_final_size=20,
                                                   prediction_task="Hf298(kcal/mol)",
                                                   save_meta=True,
                                                   save_tensors_dir=None,
                                                   meta_dir=None):
    if save_tensors_dir is not None:
        if not os.path.exists(save_tensors_dir):
            os.makedirs(save_tensors_dir)

    test_data_datasets = []
    train_datasets = []
    fidx = 0
    for host, db, table, testing_ratio in datasets:
        X, y, smis = get_data_from_db(host, db, table, prediction_task=prediction_task)

        # At this point, X just contains molecule objects,
        # so convert them to tensors. If save_tensors_dir is specified,
        # only store the file names in X and save the tensors to the disk.
        if save_tensors_dir is None:
            X = [get_molecule_tensor(mol,
                                     add_extra_atom_attribute=add_extra_atom_attribute,
                                     add_extra_bond_attribute=add_extra_bond_attribute,
                                     differentiate_atom_type=differentiate_atom_type,
                                     differentiate_bond_type=differentiate_bond_type,
                                     padding=padding,
                                     padding_final_size=padding_final_size)
                 for mol in X]
        else:
            X_new = []
            for mol in X:
                x = get_molecule_tensor(mol,
                                        add_extra_atom_attribute=add_extra_atom_attribute,
                                        add_extra_bond_attribute=add_extra_bond_attribute,
                                        differentiate_atom_type=differentiate_atom_type,
                                        differentiate_bond_type=differentiate_bond_type,
                                        padding=padding,
                                        padding_final_size=padding_final_size)
                fname = os.path.abspath(os.path.join(save_tensors_dir, '{}.npy'.format(fidx)))
                np.save(fname, x)
                X_new.append(fname)
                fidx += 1
            X = X_new

        logging.info('Splitting dataset with testing ratio of {0}...'.format(testing_ratio))
        split_data = split_test_from_train_and_val(X,
                                                   y,
                                                   smis,
                                                   shuffle_seed=0,
                                                   testing_ratio=testing_ratio)

        (X_test, y_test, X_train, y_train, smis_test, smis_train) = split_data

        test_data_datasets.append((X_test, y_test))
        train_datasets.append((X_train, y_train))

        if save_meta:
            smis_test_string = '\n'.join(smis_test)
            smis_train_string = '\n'.join(smis_train)
            if meta_dir is None:
                meta_dir = os.getcwd()
            smis_test_path = os.path.join(meta_dir, '{0}.{1}_smis_test.txt'.format(db, table))
            smis_train_path = os.path.join(meta_dir, '{0}.{1}_smis_train.txt'.format(db, table))
            with open(smis_test_path, 'w') as f_in:
                f_in.write(smis_test_string)
            with open(smis_train_path, 'w') as f_in:
                f_in.write(smis_train_string)

    # merge into one folded_Xs and folded_ys
    logging.info('Merging {} datasets for training...'.format(len(datasets)))
    (X_train, y_train) = train_datasets[0]
    if len(train_datasets) > 1:
        for X_train_1, y_train_1 in train_datasets[1:]:
            X_train.extend(X_train_1)
            y_train.extend(y_train_1)

    # merge into one X_test and y_test
    (X_test, y_test) = test_data_datasets[0]
    if len(test_data_datasets) > 1:
        for X_test_1, y_test_1 in test_data_datasets[1:]:
            X_test.extend(X_test_1)
            y_test.extend(y_test_1)

    return X_test, y_test, X_train, y_train


def prepare_folded_data_from_file(datafile,
                                  folds,
                                  add_extra_atom_attribute=True,
                                  add_extra_bond_attribute=True,
                                  differentiate_atom_type=True,
                                  differentiate_bond_type=True,
                                  padding=True,
                                  padding_final_size=20,
                                  save_tensors_dir=None,
                                  testing_ratio=0.0):
    split_data = prepare_full_train_data_from_file(datafile,
                                                   add_extra_atom_attribute=add_extra_atom_attribute,
                                                   add_extra_bond_attribute=add_extra_bond_attribute,
                                                   differentiate_atom_type=differentiate_atom_type,
                                                   differentiate_bond_type=differentiate_bond_type,
                                                   padding=padding,
                                                   padding_final_size=padding_final_size,
                                                   save_meta=False,
                                                   save_tensors_dir=save_tensors_dir,
                                                   testing_ratio=testing_ratio)
    X_test, y_test, X_train_and_val, y_train_and_val = split_data
    folded_Xs, folded_ys = prepare_folded_data(X_train_and_val, y_train_and_val, folds, shuffle_seed=2)
    return X_test, y_test, folded_Xs, folded_ys


def prepare_full_train_data_from_file(datafile,
                                      add_extra_atom_attribute=True,
                                      add_extra_bond_attribute=True,
                                      differentiate_atom_type=True,
                                      differentiate_bond_type=True,
                                      padding=True,
                                      padding_final_size=20,
                                      save_meta=True,
                                      save_tensors_dir=None,
                                      testing_ratio=0.0,
                                      meta_dir=None):
    smiles, y = [], []
    with open(datafile) as df:
        for line in df:
            line_split = line.strip().split()
            if line_split:
                smi = line_split[0]
                ysingle = [float(yi) for yi in line_split[1:]]
                smiles.append(smi)
                y.append(ysingle)
    y = np.array(y).astype(np.float32)

    logging.info('Loading data from {}...'.format(datafile))
    if save_tensors_dir is not None:
        if not os.path.exists(save_tensors_dir):
            os.makedirs(save_tensors_dir)

        X = []
        for fidx, smi in enumerate(smiles):
            mol = Molecule().fromSMILES(smi)
            x = get_molecule_tensor(mol,
                                    add_extra_atom_attribute=add_extra_atom_attribute,
                                    add_extra_bond_attribute=add_extra_bond_attribute,
                                    differentiate_atom_type=differentiate_atom_type,
                                    differentiate_bond_type=differentiate_bond_type,
                                    padding=padding,
                                    padding_final_size=padding_final_size)
            fname = os.path.abspath(os.path.join(save_tensors_dir, '{}.npy'.format(fidx)))
            np.save(fname, x)
            X.append(fname)
    else:
        X = []
        for smi in smiles:
            mol = Molecule().fromSMILES(smi)
            x = get_molecule_tensor(mol,
                                    add_extra_atom_attribute=add_extra_atom_attribute,
                                    add_extra_bond_attribute=add_extra_bond_attribute,
                                    differentiate_atom_type=differentiate_atom_type,
                                    differentiate_bond_type=differentiate_bond_type,
                                    padding=padding,
                                    padding_final_size=padding_final_size)
            X.append(x)

    logging.info('Splitting dataset with testing ratio of {}...'.format(testing_ratio))
    split_data = split_test_from_train_and_val(X,
                                               y,
                                               extra_data=smiles,
                                               shuffle_seed=0,
                                               testing_ratio=testing_ratio)

    X_test, y_test, X_train, y_train, smis_test, smis_train = split_data

    if save_meta:
        smis_test_string = '\n'.join(smis_test)
        smis_train_string = '\n'.join(smis_train)
        if meta_dir is None:
            meta_dir = os.getcwd()
        smis_test_path = os.path.join(meta_dir, 'smis_test.txt')
        smis_train_path = os.path.join(meta_dir, 'smis_train.txt')
        with open(smis_test_path, 'w') as f_in:
            f_in.write(smis_test_string)
        with open(smis_train_path, 'w') as f_in:
            f_in.write(smis_train_string)
    
    return X_test, y_test, X_train, y_train


@attr('helper')
def split_test_from_train_and_val(X, y, extra_data=None, shuffle_seed=None, testing_ratio=0.1):

    # Feed shuffle seed
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)

    # Get random number generator state so that we can shuffle multiple arrays
    rng_state = np.random.get_state()

    # Shuffle data in place
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    if extra_data is not None:
        np.random.set_state(rng_state)
        np.random.shuffle(extra_data)

    split = int(len(X) * testing_ratio)
    X_test, X_train_and_val = X[:split], X[split:]
    y_test, y_train_and_val = y[:split], y[split:]

    if extra_data is not None:
        extra_data_test, extra_data_train_and_val = extra_data[:split], extra_data[split:]

    # reset np random seed to avoid side-effect on other methods
    # relying on np.random
    if shuffle_seed is not None:
        np.random.seed()

    if extra_data is not None:
        return X_test, y_test, X_train_and_val, y_train_and_val, extra_data_test, extra_data_train_and_val
    else:
        return X_test, y_test, X_train_and_val, y_train_and_val


def prepare_folded_data(X, y, folds, shuffle_seed=None):

    # Get target size of each fold
    n = len(X)
    logging.info('Total of {} input data points'.format(n))
    target_fold_size = int(np.ceil(float(n) / folds))

    # Feed shuffle seed
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)

    # Get random number generator state so that we can shuffle multiple arrays
    rng_state = np.random.get_state()

    # Shuffle data in place
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

    # Split up data
    folded_Xs = [X[i:i+target_fold_size] for i in range(0, n, target_fold_size)]
    folded_ys = [y[i:i+target_fold_size] for i in range(0, n, target_fold_size)]

    logging.info('Split data into {} folds'.format(folds))

    # reset np random seed to avoid side-effect on other methods
    # relying on np.random
    if shuffle_seed is not None:
        np.random.seed()

    return folded_Xs, folded_ys


def split_inner_val_from_train_data(X_train, y_train, shuffle_seed=None, training_ratio=0.9):

    # Define validation set as random 10% of training
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)

    # Get random number generator state so that we can shuffle multiple arrays
    rng_state = np.random.get_state()

    # Shuffle data in place
    np.random.shuffle(X_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)

    split = int(len(X_train) * training_ratio)
    X_train, X_inner_val = X_train[:split], X_train[split:]
    y_train, y_inner_val = y_train[:split], y_train[split:]

    # reset np random seed to avoid side-effect on other methods
    # relying on np.random
    if shuffle_seed is not None:
        np.random.seed()

    return X_train, X_inner_val, y_train, y_inner_val


def prepare_data_one_fold(folded_Xs, 
                          folded_ys,
                          current_fold=0,
                          shuffle_seed=None,
                          training_ratio=0.9):

    """
    this method prepares X_train, y_train, X_inner_val, y_inner_val,
    and X_outer_val, y_outer_val
    """
    logging.info('...using fold {}'.format(current_fold+1))

    # Recombine into training and testing
    if all(isinstance(folded_X, np.ndarray) for folded_X in folded_Xs):
        X_train = np.concatenate(folded_Xs[:current_fold] + folded_Xs[(current_fold+1):])
    else:
        X_train = [x for folded_X in (folded_Xs[:current_fold] + folded_Xs[(current_fold+1):]) for x in folded_X]
    if all(isinstance(folded_y, np.ndarray) for folded_y in folded_ys):
        y_train = np.concatenate(folded_ys[:current_fold] + folded_ys[(current_fold+1):])
    else:
        y_train = [y for folded_y in (folded_ys[:current_fold] + folded_ys[(current_fold + 1):]) for y in folded_y]

    # Test is current_fold
    X_outer_val = folded_Xs[current_fold]
    y_outer_val = folded_ys[current_fold]

    X_train, X_inner_val, y_train, y_inner_val = split_inner_val_from_train_data(X_train,
                                                                                 y_train,
                                                                                 shuffle_seed,
                                                                                 training_ratio)

    logging.info('Total training: {}'.format(len(X_train)))
    logging.info('Total inner validation: {}'.format(len(X_inner_val)))
    logging.info('Total outer validation: {}'.format(len(X_outer_val)))

    return X_train, X_inner_val, X_outer_val, y_train, y_inner_val, y_outer_val
