import kenlm
import pandas as pd
import numpy as np
import torch

from datasets import load_metric
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

from pathlib import Path
from typing import Union, Dict, List, Tuple

def build_decoder(asr_processor,
                  kenlm_path: Union[str, Path],
                  alpha: float,
                  beta: float,
                  return_decoder: bool=True):
    """ Build the decoder and return either the decoder itself or the processor with LM.
    
    Parameters
    ----------
    asr_processor: Wav2Vec2Processor
        Wav2Vec2Processor instance
    kenlm_path: str or Path
        Path to trained KenLM
    alpha: float
        Alpha parameter for Decoder
    beta: float
        Beta parameter for Decoder
    return_decoder: bool
        If True, returns the decoder obtained from build_ctcdecoder method, otherwise returns Wav2Vec2ProcessorWithLM instance
        
    Returns
    ----------
        decoder or Wav2Vec2ProcessorWithLM
    """
    vocab_dict = asr_processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=kenlm_path,
        alpha=alpha,
        beta=beta
    )
    
    if return_decoder:
        return decoder
    
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=asr_processor.feature_extractor,
        tokenizer=asr_processor.tokenizer,
        decoder=decoder
    )
    
    return processor_with_lm

def grid_search_decoder(asr_processor,
                        asr_model,
                        kenlm_path: Union[str, Path],
                        alpha_beta_generator,
                        audio_sample,
                        metric=None,
                        device=None):
    if metric is None:
        metric = load_metric('wer')
    if device is None:
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    inputs = {
          'sampling_rate': audio_sample["audio"]["sampling_rate"],
          'return_tensors': "pt"
         }
        
    best_metric = -1
    result_dict = {}
    for alpha, beta in alpha_beta_generator:
        processor_with_lm = build_decoder(asr_processor,
                                          asr_model,
                                          kenlm_path,
                                          alpha=alpha,
                                          beta=beta,
                                          return_decoder=False)
        with torch.no_grad():
            ins = processor_with_lm(audio_sample["audio"]["array"], **inputs)
            logits = asr_model(**ins).logits.to(device)
        transcription = asr_processor.batch_decode(logits.cpu().numpy()).text
        metric_score = metric.compute(predictions=transcription, references=audio_sample['text'])
        result_dict[(alpha, beta)] = metric_score
        if metric_score > best_metric:
            best_metric = metric_score
            result_dict['best_metric'] = (alpha, beta)
    return result_dict