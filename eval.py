#!/usr/bin/env python3
import argparse
from concurrent.futures import process
import re
from typing import Dict

import torch
from src.data.normalization import normalize_string
from datasets import Audio, Dataset, load_dataset, load_metric
import numpy as np
import datasets

from transformers import (
    AutoFeatureExtractor,
    pipeline,
    AutoTokenizer,
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoConfig,
    AutoModelForCTC,
)


def log_results(result: Dataset, args: Dict[str, str]):
    """DO NOT CHANGE. This function computes and logs the result metrics."""

    log_outputs = args.log_outputs
    dataset_id = "_".join(args.dataset.split("/") + [args.config, args.split])

    # load metric
    wer = load_metric("wer")
    cer = load_metric("cer")

    # compute metrics
    wer_result = wer.compute(
        references=result["target"], predictions=result["prediction"]
    )
    cer_result = cer.compute(
        references=result["target"], predictions=result["prediction"]
    )

    # print & log results
    result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    with open(f"{dataset_id}_eval_results.txt", "w") as f:
        f.write(result_str)

    # log all results in text file. Possibly interesting for analysis
    if log_outputs is not None:
        pred_file = f"log_{dataset_id}_predictions.txt"
        target_file = f"log_{dataset_id}_targets.txt"

        with open(pred_file, "w") as p, open(target_file, "w") as t:

            # mapping function to write output
            def write_to_file(batch, i):
                p.write(f"{i}" + "\n")
                p.write(batch["prediction"] + "\n")
                t.write(f"{i}" + "\n")
                t.write(batch["target"] + "\n")

            result.map(write_to_file, with_indices=True)


def normalize_text(text: str, invalid_chars_regex: str, to_lower: bool) -> str:
    """
    Normalize text according to training preprocessing.
    
    'convert_numbers=True' requires the 'num2words' package
    """

    text = normalize_string(
        text, to_lower=to_lower, expand_symbols=True, convert_numbers=True
    )

    text = re.sub(invalid_chars_regex, " ", text)
    text = re.sub("\s+", " ", text).strip()

    return text


def main(args):
    # load dataset
    dataset = load_dataset(
        args.dataset, args.config, split=args.split, use_auth_token=True
    )

    # for testing: only process the first two examples as a test
    # dataset = dataset.select(range(10))
    
    if args.ctcdecode:
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.model_id)
        decoder = processor.decoder
    else:
        processor = Wav2Vec2Processor.from_pretrained(args.model_id)
        decoder = None


    # load processor
    feature_extractor = processor.feature_extractor
    sampling_rate = feature_extractor.sampling_rate
    # dataset_sampling_rate = raw_dataset[0]['audio']['sampling_rate']
    # if dataset_sampling_rate != feature_extractor.sampling_rate:
    #     print("Resampling")
    #     raw_dataset = raw_dataset.cast_column(
    #         'audio', datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #     )


    # resample audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # load eval pipeline
    if args.device is None:
        args.device = 0 if torch.cuda.is_available() else -1

    # build normalizer config
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # config = AutoConfig.from_pretrained(  args.model_id)
    model = AutoModelForCTC.from_pretrained(args.model_id)
    model.eval()
    model.to(args.device)

    # asr = pipeline(
    #     "automatic-speech-recognition",
    #     config=config,
    #     model=model,
    #     tokenizer=tokenizer,
    #     feature_extractor=feature_extractor,
    #     decoder=decoder,
    #     device=args.device,
    # )

    tokens = [
        x for x in tokenizer.convert_ids_to_tokens(range(0, tokenizer.vocab_size))
    ]
    special_tokens = [
        tokenizer.pad_token,
        tokenizer.word_delimiter_token,
        tokenizer.unk_token,
        tokenizer.bos_token,
        tokenizer.eos_token,
    ]
    non_special_tokens = [x for x in tokens if x not in special_tokens]
    invalid_chars_regex = f"[^\s{re.escape(''.join(set(non_special_tokens)))}]"
    normalize_to_lower = False
    for token in non_special_tokens:
        if token.isalpha() and token.islower():
            normalize_to_lower = True
            break

    print("Normalize to lower", normalize_to_lower)
            
    # map function to decode audio
    # def map_to_pred(
    #     batch,
    #     args=args,
    #     asr=asr,
    #     invalid_chars_regex=invalid_chars_regex,
    #     normalize_to_lower=normalize_to_lower,
    # ):
    #     prediction = asr(
    #         batch["audio"]["array"],
    #         chunk_length_s=args.chunk_length_s,
    #         stride_length_s=args.stride_length_s,
    #     )

    #     batch["prediction"] = prediction["text"]
    #     batch["target"] = normalize_text(
    #         batch["sentence"], invalid_chars_regex, normalize_to_lower
    #     )
    #     return batch
    
    def map_and_decode(batch, processor=processor, args=args):
        audio_inputs = [s['array'] for s in batch['audio']]
        sampling_rate = batch['audio'][0]['sampling_rate']
        inputs = processor(
            audio_inputs,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="longest"
        )
        
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().numpy()

        if args.ctcdecode:
            transcription = processor.batch_decode(logits, beam_width=args.beam_width).text
        else:
            transcription = processor.batch_decode(logits.argmax(-1))

        batch["prediction"] = transcription
        batch["target"] = [
            normalize_text(s, invalid_chars_regex, normalize_to_lower) for s in batch["sentence"]
        ]
        return batch

    # dataset = dataset.select(list(range(64)))
    result = dataset.map(
        map_and_decode,
        desc="Map and decode",
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=args.batch_size
    )

    #         transcription = .lower()
    # run inference on all examples
    # result = dataset.map(map_and_decode, remove_columns=dataset.column_names, batched=True, batch_size=args.batch_size)

    # compute and log_results
    # do not change function below
    log_results(result, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of the dataset. *E.g.* `'en'`  for Common Voice",
    )
    parser.add_argument(
        "--split", type=str, required=True, help="Split of the dataset. *E.g.* `'test'`"
    )
    parser.add_argument(
        "--chunk_length_s",
        type=float,
        default=None,
        help="Chunk length in seconds. Defaults to 5 seconds.",
    )
    parser.add_argument(
        "--stride_length_s",
        type=float,
        default=None,
        help="Stride of the audio chunks. Defaults to 1 second.",
    )
    parser.add_argument(
        "--log_outputs",
        action="store_true",
        help="If defined, write outputs to log file for analysis.",
    )
    parser.add_argument(
        "--ctcdecode",
        action="store_true",
        help="Apply the ctc decoder to the output (only if present in the model card).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    main(args)
