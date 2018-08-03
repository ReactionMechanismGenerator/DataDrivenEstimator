#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from dde.data import get_db_mols, str_to_mol
from dde.predictor import Predictor


def parse_command_line_arguments():
    """
    Parse the command-line arguments being passed to RMG Py. This uses the
    :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', metavar='FILE',
                        help='A file specifying which datasets to test on. Alternatively, a space-separated .csv file'
                             ' with SMILES and output(s) in the first and subsequent columns, respectively.')

    parser.add_argument('-i', '--input', metavar='FILE',
                        help='Path to predictor input file')

    parser.add_argument('-w', '--weights', metavar='H5',
                        help='Path to model weights')

    parser.add_argument('-a', '--architecture', metavar='JSON',
                        help='Path to model architecture (necessary if using uncertainty)')

    parser.add_argument('-ms', '--mean_and_std', metavar='NPZ',
                        help='Saved mean and standard deviation. '
                             'Should be loaded alongside weights if output was normalized during training')

    return parser.parse_args()
################################################################################


def read_datasets_file(datasets_file_path):
    """
    This method specify which datasets to use for validation
    """
    datasets = []
    with open(datasets_file_path, 'r') as f_in:
        for line in f_in:
            line = line.strip()
            if line and not line.startswith('#'):
                host, db, table = [token.strip() for token in line.split('.')]
                datasets.append((host, db, table))

    return datasets


def prepare_data(host, db_name, collection_name, prediction_task="Hf298(kcal/mol)"):

    # load validation data
    db_mols = get_db_mols(host, db_name, collection_name)

    smiles_list = []
    ys = []
    # decide what predict task is
    if prediction_task not in ["Hf298(kcal/mol)", "S298(cal/mol/K)", "Cp(cal/mol/K)"]:
        raise NotImplementedError("Prediction task: {0} not supported yet!".format(prediction_task))

    for i, db_mol in enumerate(db_mols):
        smiles = str(db_mol["SMILES_input"])

        if prediction_task != "Cp(cal/mol/K)":
            y = float(db_mol[prediction_task])
        else:
            Cp300 = float(db_mol["Cp300(cal/mol/K)"])
            Cp400 = float(db_mol["Cp400(cal/mol/K)"])
            Cp500 = float(db_mol["Cp500(cal/mol/K)"])
            Cp600 = float(db_mol["Cp600(cal/mol/K)"])
            Cp800 = float(db_mol["Cp800(cal/mol/K)"])
            Cp1000 = float(db_mol["Cp1000(cal/mol/K)"])
            Cp1500 = float(db_mol["Cp1500(cal/mol/K)"])
            y = np.array([Cp300, Cp400, Cp500, Cp600, Cp800, Cp1000, Cp1500])
        
        smiles_list.append(smiles)
        ys.append(y)

    return smiles_list, ys


def prepare_predictor(input_file, weights_file=None, model_file=None, mean_and_std_file=None):

    predictor = Predictor()
    predictor.load_input(input_file)
    if model_file is not None:
        predictor.load_architecture(model_file)
    predictor.load_parameters(param_path=weights_file, mean_and_std_path=mean_and_std_file)
    return predictor


def make_predictions(predictor, id_list, uncertainty=False):

    results = []
    for ident in tqdm(id_list):
        mol = str_to_mol(ident)
        result = predictor.predict(mol, sigma=uncertainty)
        results.append(result)

    return results


def evaluate(id_list, ys, results, prediction_task="Hf298(kcal/mol)", uncertainty=False):

    result_df = pd.DataFrame(index=id_list)

    result_df[prediction_task+"_true"] = pd.Series(ys, index=result_df.index)
    if uncertainty:
        ys_pred, uncertainties = zip(*results)
        result_df[prediction_task+"_uncertainty"] = pd.Series(uncertainties, index=result_df.index)
    else:
        ys_pred = results
    result_df[prediction_task+"_pred"] = pd.Series(ys_pred, index=result_df.index)

    diff = abs(result_df[prediction_task+"_true"]-result_df[prediction_task+"_pred"])
    sqe = diff ** 2.0

    # if the prediction task is Cp
    # since it has 7 values
    # we'll average them for evaluation
    if prediction_task == 'Cp(cal/mol/K)':
        diff = [np.average(d) for d in diff]
        sqe = [np.average(s) for s in sqe]

    result_df[prediction_task+"_diff"] = pd.Series(diff, index=result_df.index)
    result_df[prediction_task+"_diff_squared"] = pd.Series(sqe, index=result_df.index)

    return result_df


def display_result(result_df, prediction_task="Hf298(kcal/mol)", uncertainty=False):

    descr = result_df[prediction_task+"_diff"].describe()
    count = int(descr.loc['count'])
    mae = descr.loc['mean']

    descr = result_df[prediction_task+"_diff_squared"].describe()
    rmse = np.sqrt(descr.loc['mean'])

    print('Prediction task: {}'.format(prediction_task))
    print('Count: {}'.format(count))
    print('RMSE: {:.2f},  MAE: {:.2f}'.format(rmse, mae))

    if uncertainty:
        descr = result_df[prediction_task+"_uncertainty"].describe()
        mu = descr.loc['mean']
        print('Mean uncertainty: {:.2f}'.format(mu))
    else:
        mu = None

    return count, rmse, mae, mu


def validate(data_file, input_file, weights_file=None, model_file=None, mean_and_std_file=None):

    # load cnn predictor
    predictor = prepare_predictor(input_file, weights_file=weights_file,
                                  model_file=model_file, mean_and_std_file=mean_and_std_file)
    uncertainty = False if model_file is None else True

    if data_file.endswith('.csv'):
        id_list, ys = [], []
        with open(data_file) as df:
            for line in df:
                line_split = line.strip().split()
                if line_split:
                    smi = line_split[0]
                    y = [float(yi) for yi in line_split[1:]]
                    if len(y) == 1:
                        y = y[0]
                    id_list.append(smi)
                    ys.append(y)
        results = make_predictions(predictor, id_list, uncertainty=uncertainty)
        result_df = evaluate(id_list, ys, results,
                             prediction_task=predictor.prediction_task, uncertainty=uncertainty)
        count, rmse, mae, mu = display_result(result_df, prediction_task=predictor.prediction_task)
        evaluation_results = {data_file: {"count": count,
                                          "RMSE": rmse,
                                          "MAE": mae,
                                          "Mean uncertainty": mu}}
    else:
        datasets = read_datasets_file(data_file)

        evaluation_results = {}
        for host, db_name, collection_name in datasets:

            print("\nhost: {0}, db: {1}, collection: {2}".format(host, db_name, collection_name))

            # prepare data for testing
            smiles_list, ys = prepare_data(host, db_name, collection_name,
                                           prediction_task=predictor.prediction_task)

            # Do the predictions
            results = make_predictions(predictor, smiles_list, uncertainty=uncertainty)

            # evaluate performance
            result_df = evaluate(smiles_list, ys, results,
                                 prediction_task=predictor.prediction_task, uncertainty=uncertainty)

            # display result
            count, rmse, mae, mu = display_result(result_df, prediction_task=predictor.prediction_task)

            table = '/'.join([host, db_name, collection_name])
            evaluation_results[table] = {"count": count,
                                         "RMSE": rmse,
                                         "MAE": mae,
                                         "Mean uncertainty": mu}

    return evaluation_results


def main():

    args = parse_command_line_arguments()

    data_file = args.data
    input_file = args.input
    weights_file = args.weights
    model_file = args.architecture
    mean_and_std_file = args.mean_and_std
    evaluation_results = validate(data_file, input_file, weights_file=weights_file,
                                  model_file=model_file, mean_and_std_file=mean_and_std_file)

    with open('evaluation_results.json', 'w') as f_out:
        json.dump(evaluation_results, f_out, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
