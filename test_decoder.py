# For this demo, it is assumed that you have access to a kenlm
# language model trained on a relevant corpus to the one you are
# predicting on.
# You can create a language model using SPGISpeech by following the
# instructions in the readme in kenlm_model_creation

import argparse
import kenlm
import pandas as pd
from pyctcdecode import build_ctcdecoder
from pydub import AudioSegment
from pydub.playback import play
import random
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-k", "--kenlm", type=str, help="Path to the trained kenlm model.", required=True)
    parser.add_argument("-m", "--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models", required=True)
    parser.add_argument("-d", "--datadir", type=str, help="Path to validation dataset.", required=True)
    parser.add_argument("-l", "--datalist", type=str, help="Path to the samples tabular list in CSV containing audio path, size and transcript.", required=True)
    
    return parser.parse_args()


def greedy_decode(logits, labels):
    """Decode argmax of logits and squash in CTC fashion."""
    label_dict = {n: c for n, c in enumerate(labels)}
    prev_c = None
    out = []
    for n in logits.argmax(axis=1):
        c = label_dict.get(n, "")  # if not in labels, then assume it's ctc blank char
        if c != prev_c:
            out.append(c)
        prev_c = c
    return "".join(out)


args = get_args()

# KENLM_MODEL_LOC = "/home/raymond/demos/data/demo_huggingface_spgispeech.arpa"
# SPGI_VAL_DIR = "/data-ssd-2/speech_data/spgispeech/val/"
# SPGI_VAL_CSV = "/data-ssd-2/speech_data/spgispeech/val.csv"
KENLM_MODEL_LOC = args.kenlm
SPGI_VAL_DIR = args.datadir
SPGI_VAL_CSV = args.datalist
MODEL_NAME = args.model_name_or_path


# Load the val csv
val_df = pd.read_csv(SPGI_VAL_CSV, sep='|')

# > val_df.dtypes
# wav_filename    object
# wav_filesize     int64
# transcript      object
# dtype: object

#*****************************************************#
#______________ Todo: Customize the model ____________#
asr_processor = Wav2Vec2Processor.from_pretrained(
    MODEL_NAME)
asr_model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME)
print("Vocab: ", asr_processor.tokenizer.get_vocab())

#*****************************************************#


# Make vocab more human readable
# Replace <pad> character with placeholder '_'
# Replace '|' with ' '
# This is done for compatibility with the greedy decode function
# which is based off characters TODO @ray rewrite gd
vocab = list(asr_processor.tokenizer.get_vocab().keys())
vocab[0] = '_'
vocab[4] = ' '
# Because I find lowercase easier to read
vocab = [c.lower() for c in vocab]


decoder = build_ctcdecoder(
    labels = vocab,
    kenlm_model_path = KENLM_MODEL_LOC,
    alpha=0.6,  # tuned on a val set
    beta=2.0,  # tuned on a val set
)

continue_looping = 1

# Select random items in our val set to listen to and predict while the user desires
while continue_looping:
    input("Press Enter to select a random sample ... ")

    # select random sample
    sample_number = random.randint(0, len(val_df))
    sample_name = val_df.loc[sample_number, "wav_filename"]
    true_text = val_df.loc[sample_number, 'transcript']
    sample_loc = SPGI_VAL_DIR + sample_name

    # listen to sample
    input("Press Enter to listen to audio...")
    audio = AudioSegment.from_wav(sample_loc)
    play(audio)
    input("Press Enter to continue...")

    # play
    arr, _ = sf.read(sample_loc)

    input_values = asr_processor(arr, return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1
    logits = asr_model(input_values).logits.cpu().detach().numpy()[0]

    # get greedy decoding

    greedy_text = greedy_decode(logits, vocab)
    greedy_text = ("".join(c for c in greedy_text if c not in ["_"]))
    text = decoder.decode(logits)

    print("Sample: ", sample_name)
    print("\n")
    print("Greedy Decoding: \n" + greedy_text)
    print("\n")
    print("Language Model Decoding: \n" + text)
    print("\n")
    print("Ground truth \n" + true_text)
    print("\n")