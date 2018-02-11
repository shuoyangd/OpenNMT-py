#!/bin/sh
set -e
source /home/arenduc1/virtualenvs/py3venv/bin/activate
if [ $# -eq 3 ]
then
  echo $1 $2 $3
else
  echo "invalid number of args (expected 3)"
  exit 1
fi
DATA_TYPE=$1
RNN_SIZE=$2
TRAIN_WITH_AUG=$3
OPENMNTPATH=/home/arenduc1/virtualenvs/tools/OpenNMT-py-sding
export PYTHONPATH=$PYTHONPATH:$OPENMNTPATH/onmt/modules
PROJECT_DIR=/export/b07/arenduc1/e2e-speech
MODEL_DIR=$PROJECT_DIR/models
mkdir -p $MODEL_DIR
NAME=model.$DATA_TYPE.$RNN_SIZE
python $OPENMNTPATH/train.py -data $PROJECT_DIR/${DATA_TYPE}_data/${DATA_TYPE} -save_model $MODEL_DIR/${NAME} -encoder_type hybrid -rnn_size $RNN_SIZE -word_vec_size 123 -batch_size 20  -optim adadelta -dropout 0.1 -enc_layers 4 -dec_layers 1 -learning_rate_decay 0.99 -epochs 3 -train_with_aug $TRAIN_WITH_AUG
