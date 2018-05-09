#!/bin/bash

INPUT='predictor_input.py'
DATA_FILE='datasets.txt'
TRAIN_MODE='full_train'
BATCH_SIZE=1
NB_EPOCH=1
PATIENCE=1
source activate dde_env
export KERAS_BACKEND=theano
python ../../../scripts/train_cnn.py -i $INPUT -d ${DATA_FILE} -t ${TRAIN_MODE} -bs ${BATCH_SIZE} -ep ${NB_EPOCH} -pc ${PATIENCE}
source deactivate