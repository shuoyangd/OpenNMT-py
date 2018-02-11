#!/bin/sh
set -e
if [ $# -eq 1 ]
then
  echo $1 
else
  echo "invalid number of args (expected 1)"
  exit 1
fi
source /home/arenduc1/virtualenvs/py3venv/bin/activate
OPENMNTPATH=/home/arenduc1/virtualenvs/tools/OpenNMT-py-sding
export PYTHONPATH=$PYTHONPATH:$OPENMNTPATH/onmt/modules
PROJECT_DIR=/export/b07/arenduc1/e2e-speech
DATA_TYPE=$1
python $OPENMNTPATH/prepare_aug_data.py --folder $PROJECT_DIR/${DATA_TYPE}_data/ --prefix_list dummy.rep --prefix_out dummy
python $OPENMNTPATH/preprocess.py -train_aug_src $PROJECT_DIR/${DATA_TYPE}_data/${DATA_TYPE}.aug.train.src -train_aug_tgt $PROJECT_DIR/${DATA_TYPE}_data/${DATA_TYPE}.aug.train.tgt -train_audio_tgt $PROJECT_DIR/${DATA_TYPE}_data/${DATA_TYPE}.audio.train.tgt -valid_tgt $PROJECT_DIR/${DATA_TYPE}_data/${DATA_TYPE}.audio.valid.tgt -save_data $PROJECT_DIR/${DATA_TYPE}_data/${DATA_TYPE} -start_idx 1
