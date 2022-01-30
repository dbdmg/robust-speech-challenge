#!/bin/bash

python run_speech_recognition_ctc_bnb.py \
	--dataset_name="mozilla-foundation/common_voice_7_0" \
	--model_name_or_path="facebook/wav2vec2-xls-r-2b" \
	--dataset_config_name="it" \
	--output_dir="../wav2vec2-xls-r-2b-italian" \
	--overwrite_output_dir \
	--num_train_epochs="1" \
	--per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --gradient_accumulation_steps="2" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="sentence" \
	--length_column_name="input_length" \
	--save_steps="400" \
	--eval_steps="200" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--fp16 \
	--group_by_length \
	--push_to_hub \
    --use_auth_token \
	--hub_model_id="dbdmg/wav2vec2-xls-r-2b-italian" \
	--do_train --do_eval 