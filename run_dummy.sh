python run_speech_recognition_ctc.py \
	--dataset_name="mozilla-foundation/common_voice_7_0" \
	--model_name_or_path="hf-test/xls-r-dummy" \
	--dataset_config_name="ab" \
	--output_dir="./" \
	--overwrite_output_dir \
	--max_steps="10" \
	--per_device_train_batch_size="2" \
	--learning_rate="3e-4" \
	--save_total_limit="1" \
	--evaluation_strategy="steps" \
	--text_column_name="sentence" \
	--length_column_name="input_length" \
	--save_steps="5" \
	--layerdrop="0.0" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--fp16 \
	--group_by_length \
	--push_to_hub \
	--use_auth_token \
	--do_train --do_eval