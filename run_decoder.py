import kenlm
import argparse
import pandas as pd
import numpy as np
import random
import torch

from datasets import load_metric
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

from pathlib import Path
from typing import Union, Dict, List, Tuple
from src.decoding.decode import build_decoder, grid_search_decoder

import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-k", "--kenlm", type=str, help="Path to the trained kenlm model.", required=True)
    parser.add_argument("-m", "--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models", required=True)
    parser.add_argument("-d", "--datadir", type=str, help="Path to validation dataset.", required=True)
    parser.add_argument("-l", "--datalist", type=str, help="Path to the samples tabular list in CSV containing audio path, size and transcript.", required=True)
    
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    KENLM_MODEL_LOC = args.kenlm
    SPGI_VAL_DIR = args.datadir
    SPGI_VAL_CSV = args.datalist
    MODEL_NAME = args.model_name_or_path
    
    val_df = pd.read_csv(SPGI_VAL_CSV, sep='|')
    
    # > val_df.dtypes
    # wav_filename    object
    # wav_filesize     int64
    # transcript      object
    # dtype: object
    
    asr_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    asr_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    print("Vocab: ", asr_processor.tokenizer.get_vocab())
    
    metric = load_metric('wer')
    
    processor_with_lm = build_decoder(asr_processor,
                                      asr_model,
                                      KENLM_MODEL_LOC,
                                      alpha=.6,
                                      beta=2.0,
                                      return_decoder=False)
    for _ in range(5):
        # select random sample
        sample_number = random.randint(0, len(val_df))
        sample_name = val_df.loc[sample_number, "wav_filename"]
        true_text = val_df.loc[sample_number, 'transcript']
        sample_loc = SPGI_VAL_DIR + sample_name

        arr, _ = sf.read(sample_loc)

        inputs = {
            return_tensors: "pt",
            sampling_rate: 1600
        }

        with torch.no_grad():
            logits = processor_with_lm(arr, **inputs)
        transcription = processor.batch_decode(logits.cpu().numpy()).text
        metric_score = metric.compute(predictions=transcription, references=true_text)

        print(f'GT: {true_text}')
        print(f'Transcription: {transcription}')