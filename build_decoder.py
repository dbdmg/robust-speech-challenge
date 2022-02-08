import argparse
from transformers import AutoProcessor
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM
from huggingface_hub import Repository

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-k", "--kenlm", type=str, help="Path to the trained kenlm model.", required=True)
    parser.add_argument("-m", "--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models", required=True)
    return parser.parse_args()


def push_model_withdecoder(MODEL_NAME, KENLM_MODEL_LOC):

    """
    perché messo il my?
    """
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=KENLM_MODEL_LOC,
    )
    

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )
    
    repo = Repository(local_dir=MODEL_NAME.split("/")[1], clone_from=MODEL_NAME)
    processor_with_lm.save_pretrained(MODEL_NAME.split("/")[1])
    repo.push_to_hub(commit_message="Upload lm-boosted decoder")
    
    return
    
    
if __name__ == '__main__':
    args = parse_args()
    push_model_withdecoder(args.model_name_or_path,
                           args.kenlm
                          )