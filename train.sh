#!/bin/sh
set -e
source ./project.config
source ./run.config
#device=`free-gpu`
#echo $device
source $PYENV 
python $OPENMNTPATH/train.py -data $PROJECT_DIR/data/${DATA_TYPE}/${AUG_TYPE}/${DATA_TYPE} -save_model model -encoder_type $ARCH -rnn_size $RNN_SIZE -audio_feat_size $AUDIO_FEAT_SIZE -tgt_word_vec_size $TGT_WORD_VEC_SIZE -aug_vec_size $AUG_VEC_SIZE -batch_size $BATCH_SIZE -optim adadelta -dropout $DROPOUT -enc_layers $ENC_LAYERS -dec_layers $DEC_LAYERS -aug_enc_layers $AUG_ENC_LAYERS -learning_rate $LEARNING_RATE -learning_rate_decay 0.99 -epochs $NUM_EPOCHS -train_with_aug $TRAIN_WITH_AUG -num_concat_flags $NUM_CONCAT_FLAGS -add_noise $ADD_NOISE -use_highway_concat $HIGHWAY_CONCAT -mix_factor $MIX_FACTOR -end_mix_factor $END_MIX_FACTOR -do_subsample $SUBSAMPLE -do_weight_norm $WEIGHT_NORM -max_grad_norm $MAX_GRAD_NORM -global_attention $ATTN -input_feed $INPUT_FEED -grad_clip $GRAD_CLIP -reset_aug_iter $RESET_AUG_ITER -is_rep_aug $IS_REP_AUG
#-gpuid $device
#python /export/b07/arenduc1/OpenNMT-asr/train.py -data /export/b07/arenduc1/e2e-speech/data/chime4/dummy/chime4 -save_model /export/b07/arenduc1/e2e-speech/models/chime4.dummy.hdim.128.el.4.dl.1.al.1.aug.dummy.0.vs.83.83.83.n.0.hc.0.drop.0.3.arch.hybrid_dual_proj.sub.1.wtn.0.gn.5.gc.-1.atn.mlp.if.1.mix.0.0.0.0/model -encoder_type hybrid_dual_proj -rnn_size 128 -audio_feat_size 83 -tgt_word_vec_size 83 -aug_vec_size 83 -batch_size 64 -optim adadelta -dropout 0.3 -enc_layers 4 -dec_layers 1 -aug_enc_layers 1 -learning_rate 1.0 -learning_rate_decay 0.99 -epochs 20 -train_with_aug 0 -num_concat_flags 2 -add_noise 0 -use_highway_concat 0 -mix_factor 0.0 -end_mix_factor 0.0 -do_subsample 1 -do_weight_norm 0 -max_grad_norm 5 -grad_clip -1 -global_attention mlp -input_feed 1 -gpuid 1
