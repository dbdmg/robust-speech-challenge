import kenlm
import argparse
import pandas as pd
import numpy as np
import random
import torch
import os
import datasets

from pathlib import Path

from datasets import load_metric, load_dataset
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

from pathlib import Path
from typing import Union, Dict, List, Tuple
from src.decoding.decode import build_decoder, grid_search_decoder

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-k", "--kenlm", type=str, help="Path to the trained kenlm model.", required=True)
    parser.add_argument("-m", "--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models", required=True)
    # parser.add_argument("-d", "--datadir", type=str, help="Path to validation dataset.", required=True)
    # parser.add_argument("-l", "--datalist", type=str, help="Path to the samples tabular list in CSV containing audio path, size and transcript.", required=True)
    parser.add_argument("--dataset_name", type=str, help="The configuration name of the dataset to use (via the datasets library).", required=True)
    parser.add_argument("--dataset_config_name", type=str, help="The configuration name of the dataset to use (via the datasets library).", required=True)
    parser.add_argument("--train_split_name", type=str, help="The name of the training data set split to use (via the datasets library).", required=True)
    parser.add_argument("--use_auth_token", help="If :obj:`True`, will use the token generated when running"
            ":obj:`transformers-cli login` as HTTP bearer authorization for remote files.", required=False, default=False, action='store_true')
    
    
    return parser.parse_args()

def greedy_decode(logits, labels, ignore_set=None):
    """Decode argmax of logits and squash in CTC fashion."""
    label_dict = {n: c for n, c in enumerate(labels)}
    prev_c = None
    out = []
    for n in logits.argmax(axis=1):
        c = label_dict.get(n, "")  # if not in labels, then assume it's ctc blank char
        if not ignore_set is None and c in ignore_set:
          continue
        if c != prev_c:
            out.append(c)
        prev_c = c
    return "".join(out)

if __name__ == '__main__':
    args = parse_args()
    
    KENLM_MODEL_LOC = args.kenlm
    # SPGI_VAL_DIR = args.datadir
    # SPGI_VAL_CSV = args.datalist
    MODEL_NAME = args.model_name_or_path
    DATASET_NAME = args.dataset_name
    DATASET_CONFIG_NAME = args.dataset_config_name
    TRAIN_SPLIT_NAME = args.train_split_name
    USE_AUTH_TOKEN = args.use_auth_token
    
    # val_df = pd.read_csv(SPGI_VAL_CSV, sep='|')
    
    # > val_df.dtypes
    # wav_filename    object
    # wav_filesize     int64
    # transcript      object
    # dtype: object
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    asr_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    asr_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    print("Vocab: ", asr_processor.tokenizer.get_vocab())
    print(f'Vocab shape: {asr_processor.tokenizer.get_vocab()}')
    print(f'Loading dataset: {DATASET_NAME} - config: {DATASET_CONFIG_NAME}')
    print(f'Split: {TRAIN_SPLIT_NAME}')
    print(f'Use auth token: {USE_AUTH_TOKEN}')

    raw_dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG_NAME,
        split=TRAIN_SPLIT_NAME,
        use_auth_token=USE_AUTH_TOKEN
    )

    print(raw_dataset)

    wer_metric = load_metric('wer')
    cer_metric = load_metric('cer')

    processor_with_lm = build_decoder(asr_processor,
                                    KENLM_MODEL_LOC,
                                    alpha=.6,
                                    beta=2.0,
                                    return_decoder=False)

    feature_extractor = asr_processor.feature_extractor
    dataset_sampling_rate = raw_dataset[0]['audio']['sampling_rate']
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_dataset = raw_dataset.cast_column(
            'audio', datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    vocab_dict = asr_processor.tokenizer.get_vocab().copy()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    for idx in range(5):
        # select random sample
        # sample_number = random.randint(0, len(val_df))
        # sample_name = val_df.loc[sample_number, "wav_filename"]
        # true_text = val_df.loc[sample_number, 'transcript']
        # sample_loc = SPGI_VAL_DIR + sample_name

        arr = raw_dataset[idx]['audio']['array']
        true_text = raw_dataset[idx]['sentence'].lower()

        inputs = {
        'return_tensors': "pt",
        'sampling_rate': raw_dataset[idx]['audio']['sampling_rate']
        }

        with torch.no_grad():
            inputs = processor_with_lm(arr, **inputs)
            logits = asr_model(**inputs).logits.to(device)
        # logits = asr_model(**asr_processor(arr, **inputs)).logits.to(device)
        print(logits.shape)

        transcription_no_lm = greedy_decode(logits[0].cpu().numpy(), sorted_vocab_dict, ignore_set={'_', '[pad]', '<s>', '</s>'})
        transcription_no_lm = ("".join(c for c in transcription_no_lm if c not in ["_", '^', '$'])).replace('|', ' ')

        print('_' * 60)
        transcription_lm = processor_with_lm.batch_decode(logits.cpu().numpy()).text
        print(f'Transcription LM: {transcription_lm}')
        print(f'Transcription NO-LM: {transcription_no_lm}')
        print(f'True text: {true_text}')

        wer_lm = wer_metric.compute(predictions=transcription_lm, references=[true_text])
        wer_no_lm = wer_metric.compute(predictions=[transcription_no_lm], references=[true_text])
        print(f'LM WER: {wer_lm}')
        print(f'NO-LM WER: {wer_no_lm}')

        cer_lm = cer_metric.compute(predictions=transcription_lm, references=[true_text])
        cer_no_lm = cer_metric.compute(predictions=[transcription_no_lm], references=[true_text])
        print(f'LM CER: {cer_lm}')
        print(f'NO-LM CER: {cer_no_lm}')