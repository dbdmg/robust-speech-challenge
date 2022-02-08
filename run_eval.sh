#!/bin/bash

if [[ -z "$1" ]] ; then
    model='dbdmg/wav2vec2-xls-r-300m-italian-augmented-multids'
else
    model='dbdmg/wav2vec2-xls-r-300m-italian-augmented-multids-lm'
fi


# echo $model
python eval.py --model_id $model \
               --dataset mozilla-foundation/common_voice_7_0 \
               --split test \
               --config it \
#                --dataset speech-recognition-community-v2/dev_data \
