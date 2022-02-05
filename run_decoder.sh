#!/bin/bash

python run_decoder.py \
    --dataset_name="mozilla-foundation/common_voice_7_0" \
    --use_auth_token \
    --dataset_config_name="it" \
    --train_split_name="test[50%:52%]" \
    --model_name_or_path="dbdmg/wav2vec2-xls-r-300m-italian-augmented" \
    --kenlm "../5gram-it-cv-eos.arpa" \
    --batch_size="16" \
    --export_path="result_dict.pkl"
    