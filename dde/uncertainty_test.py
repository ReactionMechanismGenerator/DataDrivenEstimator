#!/usr/bin/env python
# -*- coding:utf-8 -*-

import unittest

import numpy as np
import keras.backend as K
from keras.layers import Input, Dense
from dde.uncertainty import RandomMask, EnsembleModel


class TestRandomMask(unittest.TestCase):
    def test_random_mask(self):
        x = Input((3,3))
        mask = RandomMask(0.5)
        y = mask(x)
        fun = K.function([x], [y])
        rng = np.random.RandomState(0)
        mask.gen_mask(rng)        
        output = fun([np.array([np.ones((3,3))])])[0][0]
        expected_output = np.array([[1,1,1],[1,0,1],[0,1,1]])
        self.assertTrue(np.array_equal(output,expected_output))
        
    def test_ensemble_model(self):
        x = Input((3,3))
        y = RandomMask(0.5)(x)
        model = EnsembleModel(input=x,output=y,seeds=[0,1])
        model.compile(loss='mse', optimizer='adam')
        y_output, std_output = model.predict([np.array([np.ones((3,3))])],sigma=True)
        expected_y_output = np.array([[0.5,1,0.5],[0.5,0,0.5],[0,0.5,0.5]])
        expected_std_output = np.array([[0.5,0,0.5],[0.5,0,0.5],[0,0.5,0.5]])
        self.assertTrue(np.array_equal(y_output[0],expected_y_output))
        self.assertTrue(np.array_equal(std_output[0],expected_std_output))
    
    def test_ensemble_train(self):
        np.random.seed(0)
        input = Input((2,))
        x = RandomMask(0.5)(input)
        y = Dense(1)(x)
        model = EnsembleModel(input=input,output=y,seeds=[0,1])
        model.compile(loss='mse', optimizer='adam')
        train_x = np.random.normal(size=(3,2))
        train_y = np.random.normal(size=(3,1))
        model.train_on_batch(train_x, train_y)
        loss = model.test_on_batch(train_x, train_y)
        y_output, std_output =model.predict(np.array([train_x[0]]),sigma=True)
        expected_y_output = 1.42774868
        expected_std_output = 0.06707561
        expected_loss = 1.07002103329
        self.assertAlmostEqual(expected_y_output,y_output[0][0],6)
        self.assertAlmostEqual(expected_std_output,std_output[0][0],6)
        self.assertAlmostEqual(expected_loss,loss,6)
