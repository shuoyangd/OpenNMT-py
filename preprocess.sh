#!/bin/sh
set -e
source ./project.config
if [ $# -eq 1 ]
then
  echo $1 
else
  echo "invalid number of args (expected 1)"
  exit 1
fi
source $PYENV #/home/arenduc1/virtualenvs/py3venv/bin/activate
#OPENMNTPATH=/home/arenduc1/virtualenvs/tools/OpenNMT-py-sding
export PYTHONPATH=$PYTHONPATH:$OPENMNTPATH/onmt/modules
#PROJECT_DIR=/export/b07/arenduc1/e2e-speech
DATA_TYPE=$1
python $OPENMNTPATH/prepare_aug_data.py --folder $PROJECT_DIR/data/${DATA_TYPE}/ --prefix_list $DATA_TYPE.noise ${DATA_TYPE}.rep --prefix_out ${DATA_TYPE}
#python $OPENMNTPATH/preprocess.py -train_aug_src $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.aug.train.src -train_aug_tgt $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.aug.train.tgt -train_audio_tgt $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.audio.train.tgt -valid_tgt $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.audio.valid.tgt -save_data $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE} -start_idx 1
