#!/bin/sh
set -e
source ./project.config
source $PYENV #/home/arenduc1/virtualenvs/py3venv/bin/activate

DATA_TYPE=$1
DATA_SECT=$2 # valid/test
AUG_TYPE=$3
#export PYTHONPATH=$PYTHONPATH:$OPENMNTPATH/onmt/modules

MODEL_DIR=$PROJECT_DIR/models/${DATA_TYPE}/${AUG_TYPE}
OUTPUTS_DIR=$PROJECT_DIR/outputs/${DATA_TYPE}/${AUG_TYPE}
mkdir -p $OUTPUTS_DIR

#looking for single word folder/best
#python $OPENMNTPATH/translate.py -model $MODEL_DIR/best -src $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.${AUG_TYPE}.audio.${DATA_SECT}.src -vocab $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.${AUG_TYPE}.vocab.pt -verbose -output $PROJECT_DIR/data/${DATA_TYPE}/${DATA_TYPE}.${AUG_TYPE}.audio.${DATA_SECT}.out -batch_size 1 -beam_size 12
for SUBSECT in "ppl" "dual.ppl" "last" "dual.last" "acc" "dual.acc"; do
  BEST_FILE=best.${SUBSECT}
  python $OPENMNTPATH/translate.py -model $MODEL_DIR/${BEST_FILE} -src $PROJECT_DIR/data/${DATA_TYPE}/${AUG_TYPE}/${DATA_TYPE}.audio.${DATA_SECT}.src -verbose -output $OUTPUTS_DIR/${BEST_FILE}.${DATA_SECT}.out -batch_size 1 -beam_size 2
  python $OPENMNTPATH/tools/per.py -p $OUTPUTS_DIR/${BEST_FILE}.${DATA_SECT}.out -r $PROJECT_DIR/data/${DATA_TYPE}/${AUG_TYPE}/${DATA_TYPE}.audio.${DATA_SECT}.tgt -m $OPENMNTPATH/tools/phones.61-39.map > $OUTPUTS_DIR/${BEST_FILE}.${DATA_SECT}.out.wer
  cat $OUTPUTS_DIR/${BEST_FILE}.${DATA_SECT}.out.wer
done
#python $OPENMNTPATH/translate.py -model /export/b18/shuoyangd/projects/asr/models/timit/best -src /export/b18/shuoyangd/projects/asr/data/timit/timit.audio.valid.src  -verbose -output tmp.out -batch_size 1 -beam_size 12
