#!/bin/sh
set -e
source ./project.config
source $PYENV #/home/arenduc1/virtualenvs/py3venv/bin/activate
if [ $# -eq 7 ]
then
  echo $1 $2 $3 $4 $5 $6 $7
else
  echo "invalid number of args (expected 7)"
  echo "DATA_TYPE RNN_SIZE TRAIN_WITH_AUG DEC_LAYERS NUM_CONCAT_FLAGS ADD_NOISE HIGHWAY_CONCAT"
  exit 1
fi
DATA_TYPE=$1
RNN_SIZE=$2
TRAIN_WITH_AUG=$3
DEC_LAYERS=$4
NUM_CONCAT_FLAGS=$5
ADD_NOISE=$6
HIGHWAY_CONCAT=$7
#OPENMNTPATH=/home/arenduc1/virtualenvs/tools/OpenNMT-py-sding
export PYTHONPATH=$PYTHONPATH:$OPENMNTPATH/onmt/modules

#PROJECT_DIR=/export/b07/arenduc1/e2e-speech
MODEL_DIR=$PROJECT_DIR/models/${DATA_TYPE}
mkdir -p $MODEL_DIR

python $OPENMNTPATH/train.py -data $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE} -save_model $MODEL_DIR/${NAME} -encoder_type hybrid -decoder_type hybrid -rnn_size $RNN_SIZE -word_vec_size 123 -batch_size 64  -optim adadelta -dropout 0.1 -enc_layers 4 -dec_layers $DEC_LAYERS -learning_rate_decay 0.99 -epochs 50 -train_with_aug $TRAIN_WITH_AUG -num_concat_flags $NUM_CONCAT_FLAGS -add_noise $ADD_NOISE -use_highway_concat $HIGHWAY_CONCAT
