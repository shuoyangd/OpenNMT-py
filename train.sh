#!/bin/sh
set -e
source ./project.config
source ./run.config
device=`free-gpu`
echo $device

source $PYENV 

python $OPENMNTPATH/train.py -data $PROJECT_DIR/data/${DATA_TYPE}/${AUG_TYPE}/${DATA_TYPE} -save_model model -encoder_type $ARCH -rnn_size $RNN_SIZE -word_vec_size $WORD_VEC_SIZE -batch_size $BATCH_SIZE -optim adadelta -dropout $DROPOUT -enc_layers 4 -dec_layers $DEC_LAYERS -aug_enc_layers $AUG_ENC_LAYERS -learning_rate $LEARNING_RATE -learning_rate_decay 0.99 -epochs $NUM_EPOCHS -train_with_aug $TRAIN_WITH_AUG -num_concat_flags $NUM_CONCAT_FLAGS -add_noise $ADD_NOISE -use_highway_concat $HIGHWAY_CONCAT -mix_factor $MIX_FACTOR -end_mix_factor $END_MIX_FACTOR -do_subsample $SUBSAMPLE -do_weight_norm $WEIGHT_NORM -max_grad_norm $MAX_GRAD_NORM -global_attention $ATTN -input_feed $INPUT_FEED -gpuid $device 
