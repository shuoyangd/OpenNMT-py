#!/bin/sh
set -e
source ./project.config
source $PYENV #/home/arenduc1/virtualenvs/py3venv/bin/activate
source ./test.config

MODEL_DIR=$PROJECT_DIR/models/${DATA_TYPE}/${AUG_TYPE}
OUTPUTS_DIR=$PROJECT_DIR/outputs/${DATA_TYPE}/${AUG_TYPE}
mkdir -p $OUTPUTS_DIR

#looking for single word folder/best
#python $OPENMNTPATH/translate.py -model $MODEL_DIR/best -src $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.${AUG_TYPE}.audio.${DATA_SECT}.src -vocab $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.${AUG_TYPE}.vocab.pt -verbose -output $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.${AUG_TYPE}.audio.${DATA_SECT}.out -batch_size 1 -beam_size 12
python $OPENMNTPATH/translate.py -model $MODEL_DIR/${BEST_FILE} -src $PROJECT_DIR/data/${DATA_TYPE}/${AUG_TYPE}/${DATA_TYPE}.audio.${DATA_SECT}.src -verbose -output $OUTPUTS_DIR/${BEST_FILE}.${DATA_SECT}.${BEAM_SIZE}.${MAX_LEN_RATIO}.out -beam_size $BEAM_SIZE -max_sent_length 10000 -max_length_ratio $MAX_LEN_RATIO -min_length_ratio $MIN_LEN_RATIO -length_prior_file $PROJECT_DIR/data/${DATA_TYPE}/${AUG_TYPE}/${DATA_TYPE}.length_prior -length_prior_factor $LENGTH_PRIOR_FACTOR
