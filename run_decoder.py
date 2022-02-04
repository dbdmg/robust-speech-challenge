import kenlm
import argparse
import pandas as pd
import numpy as np
import torch
import os
import pickle as pkl
import datasets

from pathlib import Path
from sklearn.model_selection import ParameterGrid

from datasets import load_metric, load_dataset
from datasets import Dataset, DatasetDict, Metric, IterableDatasetDict, IterableDataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, PreTrainedTokenizer

from pyctcdecode import build_ctcdecoder, BeamSearchDecoderCTC
from pyctcdecode.language_model import load_unigram_set_from_arpa, LanguageModel, AbstractLanguageModel
from pyctcdecode.alphabet import Alphabet, verify_alphabet_coverage

from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Union, Dict, List, Tuple, Optional, Collection
from functools import partial
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
    parser.add_argument('--batch_size', help="Batch size", type=int, required=True)
    parser.add_argument('--export_path', help="Path to save the resulting dictionary", type=str, required=True)
    
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

def tokenize(sample, tokenizer, feature_extractor):
    if isinstance(sample, datasets.arrow_dataset.Batch):
        sentence_inputs = [s.lower() for s in sample['sentence']]
        audio_inputs = [s['array'] for s in sample['audio']]
        sampling_rate = sample['audio'][0]['sampling_rate']
    else:
        sentence_inputs = sample['sentence'].lower()
        audio_inputs = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']

    
    if tokenizer:
        token = tokenizer(sentence_inputs,
                        padding='longest')
        token['sentence_attention_mask'] = token.pop('attention_mask')
    else:
        token = {}

    audio = feature_extractor(audio_inputs,
                                sampling_rate=sampling_rate,
                                padding='longest')
    audio['audio_attention_mask'] = audio.pop('attention_mask')
    
    return dict(**audio, **token)

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
    BATCH_SIZE = args.batch_size
    EXPORT_PATH = args.export_path

    alpha_beta_gen = ParameterGrid({
        'alpha': [0.5, 0.6, 0.7, 0.8],
        'beta': [1.0, 2.0, 3.0, 4.0]
    })
    
    # val_df = pd.read_csv(SPGI_VAL_CSV, sep='|')
    
    # > val_df.dtypes
    # wav_filename    object
    # wav_filesize     int64
    # transcript      object
    # dtype: object
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    asr_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    asr_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
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

    feature_extractor = asr_processor.feature_extractor
    dataset_sampling_rate = raw_dataset[0]['audio']['sampling_rate']
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_dataset = raw_dataset.cast_column(
            'audio', datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # vocab_dict = asr_processor.tokenizer.get_vocab().copy()
    # sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    remove_columns = [
        'accent','age', 'path', 'client_id',
        'down_votes', 'up_votes', 'gender',
        'locale', 'segment', 'audio', 'sentence'
    ]

    processed_dataset = raw_dataset.map(partial(tokenize,
                                                tokenizer=asr_processor.tokenizer,
                                                feature_extractor=asr_processor.feature_extractor),
                                        remove_columns=remove_columns,
                                        batched=True,
                                        batch_size=BATCH_SIZE)
    processed_dataset.set_format(type='torch', columns=['audio_attention_mask', 'input_values', 'sentence_attention_mask', 'input_ids'])

    loader = DataLoader(processed_dataset, batch_size=BATCH_SIZE)

    result_dict = grid_search_decoder(asr_processor,
                                        asr_model,
                                        KENLM_MODEL_LOC,
                                        alpha_beta_gen,
                                        loader,
                                        store_all_results=False)

    with open(EXPORT_PATH, 'wb') as fp:
        pkl.dump(result_dict, fp)