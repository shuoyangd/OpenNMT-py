#!/bin/sh
set -e
source ./project.config
source $PYENV #/home/arenduc1/virtualenvs/py3venv/bin/activate

DATA_TYPE=$1
DATA_SECT=$2 # valid/test
export PYTHONPATH=$PYTHONPATH:$OPENMNTPATH/onmt/modules

MODEL_DIR=$PROJECT_DIR/models/${DATA_TYPE}
mkdir -p $MODEL_DIR

python $OPENMNTPATH/translate.py -model $MODEL_DIR/best -src $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.audio.${DATA_SECT}.src -vocab $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.vocab.pt -verbose -output $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.audio.${DATA_SECT}.out -batch_size 1 -beam_size 12
