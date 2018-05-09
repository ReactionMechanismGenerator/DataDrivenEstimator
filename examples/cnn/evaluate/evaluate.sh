#!/bin/bash

CNN_MODEL='test_model'
DATA_FILE='test_datasets.txt'
source activate rmg_env
export KERAS_BACKEND=theano
python ../../../scripts/evaluate_cnn.py -d ${DATA_FILE} -m ${CNN_MODEL}
source deactivate