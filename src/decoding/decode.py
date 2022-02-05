import datasets
import kenlm
import numpy as np
import torch

from datasets import load_metric
from datasets import Dataset, DatasetDict, Metric, IterableDatasetDict, IterableDataset

from pyctcdecode import build_ctcdecoder, BeamSearchDecoderCTC
from pyctcdecode.language_model import AbstractLanguageModel, LanguageModel, load_unigram_set_from_arpa
from pyctcdecode.alphabet import Alphabet, verify_alphabet_coverage

from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, PreTrainedTokenizer

from pathlib import Path
from collections import defaultdict
from typing import Union, Dict, List, Tuple, Optional, Collection

def get_kenlm_model_unigrams(kenlm_model_path: str,
                             return_alphabet: bool=False,
                             labels=None) -> Tuple[kenlm.Model, Collection[str]]:
    kenlm_model = kenlm.Model(kenlm_model_path)
    if kenlm_model_path.endswith(".arpa"):
        unigrams = load_unigram_set_from_arpa(kenlm_model_path)
    else:
        print(
            "Unigrams not provided and cannot be automatically determined from LM file (only "
            "arpa format). Decoding accuracy might be reduced."
        )
        unigrams = None
    if not return_alphabet:
        return kenlm_model, unigrams
    else:
        return kenlm_model, unigrams, Alphabet.build_alphabet(labels)

def my_build_ctc_decoder(
        labels: List[str],
        kenlm_model: kenlm.Model,
        unigrams: Collection[str],
        alpha: float = 0.5,
        beta: float = 1.5,
        unk_score_offset: float = -10.0,
        lm_score_boundary: bool = True,
        alphabet: Optional[Alphabet]=None) -> BeamSearchDecoderCTC:
    if alphabet is None:
        alphabet = Alphabet.build_alphabet(labels)
    if unigrams is not None:
        verify_alphabet_coverage(alphabet, unigrams)
    if kenlm_model is not None:
        language_model: Optional[AbstractLanguageModel] = LanguageModel(
            kenlm_model,
            unigrams,
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            score_boundary=lm_score_boundary,
        )
    else:
        language_model = None
    return BeamSearchDecoderCTC(alphabet, language_model)

def generate_labels(vocab_dict, sort=True):
    if sort:
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    else:
        sorted_vocab_dict = vocab_dict

    return list(sorted_vocab_dict.keys())

def build_decoder(asr_processor,
                  kenlm_path: Union[str, Path, kenlm.Model],
                  alpha: float,
                  beta: float,
                  return_decoder: bool=True,
                  unigrams: Optional[Collection[str]]=None,
                  alphabet=None,
                  **kwargs) -> Union[BeamSearchDecoderCTC, Wav2Vec2ProcessorWithLM]:
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
    labels = generate_labels(asr_processor.tokenizer.get_vocab(), sort=True)

    if not unigrams is None and isinstance(kenlm_path, kenlm.Model):
        decoder = my_build_ctc_decoder(
                labels,
                kenlm_path,
                unigrams,
                alpha,
                beta,
                alphabet=alphabet,
                **kwargs
        )
    else:
        assert isinstance(kenlm_path, (str, Path))
        decoder = build_ctcdecoder(
                labels,
                kenlm_path,
                alpha=alpha,
                beta=beta,
                **kwargs
        )
    
    if return_decoder:
        return decoder
    
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=asr_processor.feature_extractor,
        tokenizer=asr_processor.tokenizer,
        decoder=decoder
    )
    
    return processor_with_lm

def _generate_key(args: dict) -> Tuple:
    k = tuple(v for _, v in sorted(args.items(), key=lambda it: it[0]))
    return k

def compute_total_len(dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, DataLoader],
                      gt_text_name: str='sentence',
                      tokenizer: Optional[PreTrainedTokenizer]=None) -> int:
    res = 0
    if not isinstance(dataset, DataLoader):
        for audio_sample in dataset:
            res += len(audio_sample[gt_text_name])
    else:
        for b in dataset:
            decoded = tokenizer.batch_decode(b['input_ids'], skip_special_tokens=True)
            for phrase in decoded:
                res += len(phrase)

    return res

def grid_search_decoder(asr_processor: Wav2Vec2Processor,
                        asr_model: Wav2Vec2ForCTC,
                        kenlm_path: Union[str, Path],
                        decoder_param_generator,
                        loader: DataLoader,
                        metric: Optional[Union[Metric, Dict]]=None,
                        device=None,
                        store_all_results: bool=False) -> Dict[Tuple[float, float], Dict[str, np.ndarray]]:
    if metric is None:
        metric = {
            'wer': load_metric('wer'),
            'cer': load_metric('cer')
        }
    elif isinstance(metric, datasets.Metric):
        metric = {
            metric.name: metric
        }
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asr_model = asr_model.to(device)
    result_dict = defaultdict(lambda: defaultdict(list if store_all_results else float))
    if not store_all_results:
        # pre-compute total length
        total_len = compute_total_len(loader, tokenizer=asr_processor.tokenizer)
        print(f'Total length: {total_len}')

    kenlm_model, unigrams, alphabet = get_kenlm_model_unigrams(kenlm_path,
                                                                return_alphabet=True,
                                                                labels=generate_labels(asr_processor.tokenizer.get_vocab()))

    decoders = {
        _generate_key(decoder_params): build_decoder(asr_processor,
                                                    kenlm_model,
                                                    return_decoder=False,
                                                    unigrams=unigrams,
                                                    **decoder_params)
        for decoder_params in decoder_param_generator
    }
    for idy, audio_batch in enumerate(loader):
        if idy % 10 == 0:
            print(f'Evaluating sample {idy + 1}/{len(loader)}')

        true_text = [t.lower() for t in asr_processor.batch_decode(audio_batch['input_ids'],
                                                                skip_special_tokens=True)]
        true_len = [len(t) for t in true_text]
        sum_true_len = 0
        for tl in true_len:
            sum_true_len += tl

        with torch.no_grad():
            audio_dev = audio_batch['input_values'].to(device)
            audio_att_mask = audio_batch['audio_attention_mask'].to(device)
            logits = asr_model(input_values=audio_dev,
                                attention_mask=audio_att_mask).logits.to(device)
        for idx, decoder_params in enumerate(decoder_param_generator):
            if idx % 10 == 0:
                print(f'Evaluating config {idx + 1}/{len(decoder_param_generator)}')
            k = _generate_key(decoder_params)
            decoder = decoders[k]
            transcription = decoder.batch_decode(logits.detach().cpu().numpy()).text
        
            for m in metric:
                metric_score = metric[m].compute(predictions=transcription, references=true_text)
                if store_all_results:
                    result_dict[k][m].append(metric_score)
                    result_dict[k]['weight'].append(sum_true_len)
                else:
                    result_dict[k][m] += metric_score * sum_true_len / total_len

    for k in result_dict:
        result_dict[k] = dict(result_dict[k])
        if store_all_results:
            for m in result_dict[k]:
                result_dict[k][m] = np.ndarray(result_dict[k][m])
    return dict(result_dict)