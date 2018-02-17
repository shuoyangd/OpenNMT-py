#!/bin/sh
set -e
source ./project.config
RUN_CONFG=$1
source ./${RUN_CONFG}

source $PYENV 

MODEL_DIR=$PROJECT_DIR/models/${DATA_TYPE}/${AUG_TYPE}
mkdir -p $MODEL_DIR

NAME=model.$DATA_TYPE.hdim.$RNN_SIZE.dl.$DEC_LAYERS.aug.$AUG_TYPE.$TRAIN_WITH_AUG.n.$ADD_NOISE.hc.$HIGHWAY_CONCAT.drop.$DROPOUT.arch.$ARCH.sub.$SUBSAMPLE.wtn.$WEIGHT_NORM
echo $NAME

python $OPENMNTPATH/train.py -data $PROJECT_DIR/data/${DATA_TYPE}/${AUG_TYPE}/${DATA_TYPE} -save_model $MODEL_DIR/${NAME} -encoder_type $ARCH -rnn_size $RNN_SIZE -word_vec_size $WORD_VEC_SIZE -batch_size $BATCH_SIZE -optim adadelta -dropout $DROPOUT -enc_layers 4 -dec_layers $DEC_LAYERS -learning_rate_decay 0.99 -epochs $NUM_EPOCHS -train_with_aug $TRAIN_WITH_AUG -num_concat_flags $NUM_CONCAT_FLAGS -add_noise $ADD_NOISE -use_highway_concat $HIGHWAY_CONCAT -mix_factor $MIX_FACTOR  -end_mix_factor $END_MIX_FACTOR -do_subsample $SUBSAMPLE -do_weight_norm $WEIGHT_NORM
