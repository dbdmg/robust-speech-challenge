#!/bin/bash

if [[ -z "$1" ]] ; then
    model='dbdmg/wav2vec2-xls-r-300m-italian-augmented-multids'
else
    model='dbdmg/wav2vec2-xls-r-300m-italian-augmented-multids-lm'
fi


# echo $model
python eval.py --model_id $model \
               --dataset speech-recognition-community-v2/dev_data \
               --config it \
               --split validation \
               --chunk_length_s 5.0 \
               --stride_length_s 1.0 \
