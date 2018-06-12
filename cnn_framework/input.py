#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import logging
from cnn_framework.cnn_model import build_model
from cnn_framework.molecule_tensor import get_attribute_vector_size

predictor = None


def predictor_model(prediction_task="Hf298(kcal/mol)",
                    embedding_size=512, attribute_vector_size=None, depth=2, 
                    add_extra_atom_attribute=True, add_extra_bond_attribute=True,
                    differentiate_atom_type=True,
                    differentiate_bond_type=True,
                    scale_output=0.05, 
                    padding=False, padding_final_size=20,
                    mol_conv_inner_activation='tanh',
                    mol_conv_outer_activation='softmax',
                    hidden=0, hidden_depth=1, hidden_activation='tanh',
                    output_activation='linear', output_size=1, 
                    lr=0.01, optimizer='adam', loss='mse',
                    dropout_rate_inner=0.0, dropout_rate_outer=0.0,
                    dropout_rate_hidden=0.0, dropout_rate_output=0.0,
                    n_model=None):
                    
    if dropout_rate_inner==0.0 and dropout_rate_outer==0.0 \
        and dropout_rate_hidden==0.0 and dropout_rate_output==0.0:
        n_model=None
    
    if attribute_vector_size is None:
        attribute_vector_size = get_attribute_vector_size(add_extra_atom_attribute, 
                                                          add_extra_bond_attribute,
                                                          differentiate_atom_type,
                                                          differentiate_bond_type)
    
    model = build_model(embedding_size, attribute_vector_size, depth,
                        scale_output, padding,
                        mol_conv_inner_activation,
                        mol_conv_outer_activation,
                        hidden, hidden_depth, hidden_activation,
                        output_activation, output_size,
                        lr, optimizer, loss,
                        dropout_rate_inner, dropout_rate_outer,
                        dropout_rate_hidden, dropout_rate_output,
                        n_model, padding_final_size)
    
    predictor.prediction_task = prediction_task
    predictor.model = model
    predictor.add_extra_atom_attribute = add_extra_atom_attribute
    predictor.add_extra_bond_attribute = add_extra_bond_attribute
    predictor.differentiate_bond_type = differentiate_bond_type
    predictor.differentiate_atom_type = differentiate_atom_type
    predictor.padding = padding
    predictor.padding_final_size = padding_final_size


def read_input_file(path, predictor0):

    global predictor
    
    full_path = os.path.abspath(os.path.expandvars(path))
    try:
        f = open(full_path)
    except IOError as e:
        logging.error('The input file "{0}" could not be opened.'.format(full_path))
        logging.info('Check that the file exists and that you have read access.')
        raise e

    logging.info('Reading predictor input file "{0}"...'.format(full_path))
    logging.info(f.read())
    f.seek(0)  # return to beginning of file

    predictor = predictor0
    
    global_context = {'__builtins__': None}
    local_context = {
        '__builtins__': None,
        'True': True,
        'False': False,
        'predictor_model': predictor_model,
    }

    try:
        exec f in global_context, local_context
    except (NameError, TypeError, SyntaxError) as e:
        logging.error('The input file "{0}" was invalid:'.format(full_path))
        logging.exception(e)
        raise
    finally:
        f.close()

    logging.info('')
