import librosa
import pysrt
from datetime import datetime
from typing import List, Tuple
import numpy as np

def get_index(t: float, sr: int) -> int:
    return int(t * sr)

def total_seconds(t: datetime) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def extract_chunks(audio_file: str, srt_file: str) -> Tuple[List[Tuple[np.ndarray, str]], int]:
    """Split an audio file into chunks, based on its subtitles.
    
    Parameters
    ----------
    audio_file : str
        the path to the audio file (must be a format supported by librosa.load)
    srt_file : str
        the path to the subtitles file (must be a .srt file)

    Returns
    -------
    chunks : list of tuples (sequence, subtitle)
        a list with an item for each piece of subtitles included in srt_file. For 
        each subtitle, the audio sequence (as a numpy array) and the text of
        the subtitles are returned
    """
    srt = pysrt.open(srt_file)
    seq, sr = librosa.load(audio_file)

    chunks = []

    for sub in srt:
        ts_start = total_seconds(sub.start.to_time())
        ts_end = total_seconds(sub.end.to_time())

        ndx_start = get_index(ts_start, sr)
        ndx_end = get_index(ts_end, sr)

        # NOTE: the subtitle may require some further processing
        # (e.g. to remove newlines, brackets, etc.)
        chunks.append((seq[ndx_start:ndx_end], sub.text))
    return chunks, sr
