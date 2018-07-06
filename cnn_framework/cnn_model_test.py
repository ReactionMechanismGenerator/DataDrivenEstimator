#!/usr/bin/env python
# -*- coding:utf-8 -*-

from cnn_framework.cnn_model import build_model, save_model
from cnn_framework.layers import MoleculeConv
from cnn_framework.uncertainty import RandomMask, EnsembleModel
from keras.layers.core import Dense
import unittest
import cnn_framework
import os


class TestCNNModel(unittest.TestCase):

    def test_build_model(self):

        embedding_size = 300
        attribute_vector_size = 10
        hidden = 0
        test_model = build_model(embedding_size=embedding_size,
                                 attribute_vector_size=attribute_vector_size,
                                 hidden=hidden
                                 )
        self.assertEqual(len(test_model.layers), 3)
        self.assertTrue(isinstance(test_model.layers[1], MoleculeConv))
        self.assertTrue(isinstance(test_model.layers[2], Dense))

        self.assertEqual(test_model.layers[1].inner_dim, attribute_vector_size-1)
        self.assertEqual(test_model.layers[1].units, embedding_size)

    def test_save_model(self):

        embedding_size = 300
        attribute_vector_size = 10
        hidden = 0
        test_model = build_model(embedding_size=embedding_size,
                                 attribute_vector_size=attribute_vector_size,
                                 hidden=hidden
                                 )

        save_model_folder = os.path.join(os.path.dirname(cnn_framework.__file__),
                                         'test_data',
                                         'save_model_test')
        if not os.path.exists(save_model_folder):
            os.mkdir(save_model_folder)

        fpath = os.path.join(save_model_folder, 'model')

        save_model(test_model, [1.0], [1.0], 1.0, 1.0, fpath)

        import shutil
        shutil.rmtree(save_model_folder)
        
    def test_build_ensemble_model(self):

        embedding_size = 300
        attribute_vector_size = 10
        hidden = 0
        n_model = 10
        dropout_rate_inner = 0.5
        dropout_rate_outer = 0.5
        dropout_rate_hidden = 0.5
        dropout_rate_output = 0.5
        padding_final_size = 50
        test_model = build_model(embedding_size = embedding_size,
                                 attribute_vector_size = attribute_vector_size,
                                 hidden = hidden,
                                 n_model = n_model,
                                 dropout_rate_inner=dropout_rate_inner, 
                                 dropout_rate_outer=dropout_rate_outer,
                                 dropout_rate_hidden=dropout_rate_hidden, 
                                 dropout_rate_output=dropout_rate_output,
                                 padding_final_size = padding_final_size
                                 )
        self.assertEqual(len(test_model.seeds), n_model)
        self.assertTrue(isinstance(test_model, EnsembleModel))
        self.assertTrue(isinstance(test_model.layers[1], MoleculeConv))
        self.assertTrue(isinstance(test_model.layers[2], RandomMask))
        self.assertTrue(isinstance(test_model.layers[3], Dense))
