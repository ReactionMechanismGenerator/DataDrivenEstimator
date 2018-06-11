#!/usr/bin/env python
# -*- coding:utf-8 -*-

from cnn_framework.predictor import Predictor
from cnn_framework.layers import MoleculeConv
from keras.layers.core import Dense
import unittest
import os
import shutil
import numpy as np
import cnn_framework
from rmgpy.molecule import Molecule


class TestPredictor(unittest.TestCase):

    def setUp(self):

        self.predictor = Predictor()

    def test_model(self):

        self.predictor.build_model()
        predictor_model = self.predictor.model
        self.assertEqual(len(predictor_model.layers), 3)
        self.assertTrue(isinstance(predictor_model.layers[0], MoleculeConv))
        self.assertTrue(isinstance(predictor_model.layers[1], Dense))

        self.assertEqual(predictor_model.layers[0].inner_dim, 32)
        self.assertEqual(predictor_model.layers[0].units, 512)

    def test_load_input(self):

        test_predictor_input = os.path.join(os.path.dirname(cnn_framework.__file__),
                                            'test_data',
                                            'minimal_predictor',
                                            'predictor_input.py'
                                            )
        self.predictor.load_input(test_predictor_input)

        predictor_model = self.predictor.model
        self.assertEqual(len(predictor_model.layers), 3)
        self.assertTrue(isinstance(predictor_model.layers[0], MoleculeConv))
        self.assertTrue(isinstance(predictor_model.layers[1], Dense))
        self.assertTrue(isinstance(predictor_model.layers[2], Dense))

        gfp = self.predictor.model.layers[0]
        dense1 = self.predictor.model.layers[1]
        dense2 = self.predictor.model.layers[2]

        self.assertEqual(gfp.W_inner.shape.eval()[0], 4)
        self.assertEqual(gfp.W_inner.shape.eval()[1], 38)
        self.assertEqual(gfp.W_inner.shape.eval()[2], 38)
        self.assertEqual(gfp.b_inner.shape.eval()[0], 4)
        self.assertEqual(gfp.b_inner.shape.eval()[1], 1)
        self.assertEqual(gfp.b_inner.shape.eval()[2], 38)

        self.assertEqual(gfp.W_output.shape.eval()[0], 4)
        self.assertEqual(gfp.W_output.shape.eval()[1], 38)
        self.assertEqual(gfp.W_output.shape.eval()[2], 512)
        self.assertEqual(gfp.b_output.shape.eval()[0], 4)
        self.assertEqual(gfp.b_output.shape.eval()[1], 1)
        self.assertEqual(gfp.b_output.shape.eval()[2], 512)

        self.assertEqual(dense1.W.shape.eval()[0], 512)
        self.assertEqual(dense1.W.shape.eval()[1], 50)
        self.assertEqual(dense1.b.shape.eval()[0], 50)

        self.assertEqual(dense2.W.shape.eval()[0], 50)
        self.assertEqual(dense2.W.shape.eval()[1], 1)
        self.assertEqual(dense2.b.shape.eval()[0], 1)

    def test_specify_datasets(self):
        """
        Test the datasets specification is done properly
        """
        datasets_file = os.path.join(os.path.dirname(cnn_framework.__file__),
                                     'test_data',
                                     'minimal_predictor',
                                     'datasets.txt')
        self.predictor.specify_datasets(datasets_file)
        expected_datasets = [('rmg', 'sdata134k', 'polycyclic_2954_table', 0.1),
                             ('rmg', 'sdata134k', 'cyclic_O_only_table', 0.1)]

        self.assertEqual(self.predictor.datasets, expected_datasets)

    def test_load_parameters(self):

        test_predictor_input = os.path.join(os.path.dirname(cnn_framework.__file__),
                                            'test_data',
                                            'minimal_predictor',
                                            'predictor_input.py')
        self.predictor.load_input(test_predictor_input)

        param_path = os.path.join(os.path.dirname(cnn_framework.__file__),
                                  'test_data',
                                  'minimal_predictor',
                                  'weights.h5')
        self.predictor.load_parameters(param_path)

        gfp = self.predictor.model.layers[0]
        dense1 = self.predictor.model.layers[1]
        dense2 = self.predictor.model.layers[2]

        self.assertAlmostEqual(gfp.W_inner.eval()[0][0][0], 1.000, 3)
        self.assertAlmostEqual(gfp.b_inner.eval()[0][0][0], 0.000, 3)
        self.assertAlmostEqual(gfp.W_output.eval()[0][0][0], 0.040, 3)
        self.assertAlmostEqual(gfp.b_output.eval()[0][0][0], -0.561, 3)

        self.assertAlmostEqual(dense1.W.eval()[0][0], -0.023, 3)
        self.assertAlmostEqual(dense1.b.eval()[0], 1.517, 3)

        self.assertAlmostEqual(dense2.W.eval()[0][0], -4.157, 3)
        self.assertAlmostEqual(dense2.b.eval()[0], 1.515, 3)

    def test_predict(self):
        """
        Test predictor is predicting within a reasonable range
        we should change weights.h5 every time we change feature space
        """

        test_predictor_input = os.path.join(os.path.dirname(cnn_framework.__file__),
                                            'test_data',
                                            'minimal_predictor',
                                            'predictor_input.py')
        self.predictor.load_input(test_predictor_input)
        self.assertTrue(self.predictor.add_extra_atom_attribute)
        self.assertTrue(self.predictor.add_extra_bond_attribute)

        param_path = os.path.join(os.path.dirname(cnn_framework.__file__),
                                  'test_data',
                                  'minimal_predictor',
                                  'weights.h5')
        self.predictor.load_parameters(param_path)

        mol_test = Molecule().fromAdjacencyList("""1  C u0 p0 c0 {2,B} {6,B} {7,S}
2  C u0 p0 c0 {1,B} {3,B} {8,S}
3  C u0 p0 c0 {2,B} {4,B} {9,S}
4  C u0 p0 c0 {3,B} {5,B} {10,S}
5  C u0 p0 c0 {4,B} {6,B} {11,S}
6  C u0 p0 c0 {1,B} {5,B} {12,S}
7  H u0 p0 c0 {1,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {4,S}
11 H u0 p0 c0 {5,S}
12 H u0 p0 c0 {6,S}
""")

        h298_predicted = self.predictor.predict(mol_test)

        self.assertAlmostEqual(h298_predicted, 19.5, 0)

    def test_normalize(self):
        y1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        y2 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y1_norm_expected = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        y2_norm_expected = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

        y1_norm, y2_norm = self.predictor.normalize_output(y1, y2)

        self.assertAlmostEqual(self.predictor.y_mean, 4.0)
        self.assertAlmostEqual(self.predictor.y_std, 2.0)
        self.assertTrue(np.allclose(y1_norm, y1_norm_expected))
        self.assertTrue(np.allclose(y2_norm, y2_norm_expected))

        y1 = [[1.0, 2.0], [3.0, 4.0]]
        y2 = [[2.0, 3.0], [4.0, 5.0]]
        mean_expected = [2.0, 3.0]
        std_expected = [1.0, 1.0]
        y1_norm_expected = [[-1.0, -1.0], [1.0, 1.0]]
        y2_norm_expected = [[0.0, 0.0], [2.0, 2.0]]

        y1_norm, y2_norm = self.predictor.normalize_output(y1, y2)

        self.assertTrue(np.allclose(self.predictor.y_mean, mean_expected))
        self.assertTrue(np.allclose(self.predictor.y_std, std_expected))
        self.assertTrue(np.allclose(y1_norm, y1_norm_expected))
        self.assertTrue(np.allclose(y2_norm, y2_norm_expected))

        self.predictor.y_mean = None
        self.predictor.y_std = None

    def test_kfcv_train(self):
        test_predictor_input = os.path.join(os.path.dirname(cnn_framework.__file__),
                                            'test_data',
                                            'minimal_predictor',
                                            'predictor_input.py')
        self.predictor.load_input(test_predictor_input)
        param_path = os.path.join(os.path.dirname(cnn_framework.__file__),
                                  'test_data',
                                  'minimal_predictor',
                                  'weights.h5')
        self.predictor.load_parameters(param_path)

        datafile = os.path.join(os.path.dirname(cnn_framework.__file__),
                                'test_data',
                                'datafile.csv')
        self.predictor.data_file = datafile
        self.predictor.get_data_from_file = True

        out_dir = os.path.join(os.path.dirname(cnn_framework.__file__),
                               'test_data',
                               'test_out')
        self.predictor.out_dir = out_dir
        save_model_path = os.path.join(out_dir, 'saved_model')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)

        lr_func = "float({0} * np.exp(- epoch / {1}))".format(0.0007, 30.0)
        self.predictor.kfcv_train(2, lr_func, save_model_path, nb_epoch=2, patience=-1, testing_ratio=0.1)

        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'best_model.h5')))  # Kind of lucky this exists
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'current_model.h5')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_0.h5')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_0.hist')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_0.json')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_0.png')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_0_loss_report.txt')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_1.h5')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_1.hist')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_1.json')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_1.png')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'fold_1_loss_report.txt')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'full_folds_loss_report.txt')))

        shutil.rmtree(out_dir)

    def test_full_train(self):
        self.predictor.normalize = True

        test_predictor_input = os.path.join(os.path.dirname(cnn_framework.__file__),
                                            'test_data',
                                            'minimal_predictor',
                                            'predictor_input.py')
        self.predictor.load_input(test_predictor_input)

        datafile = os.path.join(os.path.dirname(cnn_framework.__file__),
                                'test_data',
                                'datafile.csv')
        self.predictor.data_file = datafile
        self.predictor.get_data_from_file = True

        out_dir = os.path.join(os.path.dirname(cnn_framework.__file__),
                               'test_data',
                               'test_out')
        self.predictor.out_dir = out_dir
        save_model_path = os.path.join(out_dir, 'saved_model')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)

        lr_func = "float({0} * np.exp(- epoch / {1}))".format(0.0007, 30.0)
        self.predictor.full_train(lr_func, save_model_path, nb_epoch=2, training_ratio=1.0, testing_ratio=0.0)

        self.assertTrue(os.path.exists(os.path.join(out_dir, 'smis_test.txt')))
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'smis_train.txt')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'current_model.h5')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'full_train.h5')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'full_train.hist')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'full_train.json')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'full_train.png')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'full_train_loss_report.txt')))
        self.assertTrue(os.path.exists(os.path.join(save_model_path, 'full_train_mean_std.npz')))

        self.predictor.normalize = False
        shutil.rmtree(out_dir)

    def test_kfcv_batch_train(self):
        test_predictor_input = os.path.join(os.path.dirname(cnn_framework.__file__),
                                            'test_data',
                                            'minimal_predictor',
                                            'predictor_input.py')
        self.predictor.load_input(test_predictor_input)

        datafile = os.path.join(os.path.dirname(cnn_framework.__file__),
                                'test_data',
                                'datafile.csv')
        self.predictor.data_file = datafile
        self.predictor.get_data_from_file = True

        out_dir = os.path.join(os.path.dirname(cnn_framework.__file__),
                               'test_data',
                               'test_out')
        self.predictor.out_dir = out_dir
        save_model_path = os.path.join(out_dir, 'saved_model')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)

        weights_file = os.path.join(os.path.dirname(cnn_framework.__file__),
                                    'test_data',
                                    'minimal_predictor',
                                    'weights.h5')

        self.predictor.kfcv_batch_train(3, pretrained_weights=weights_file,
                                        batch_size=2, nb_epoch=2, training_ratio=0.8, testing_ratio=0.1)

        self.assertTrue(os.path.exists(os.path.join(out_dir, 'history.json_fold_0')))
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'history.json_fold_1')))
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'history.json_fold_2')))

        shutil.rmtree(out_dir)
