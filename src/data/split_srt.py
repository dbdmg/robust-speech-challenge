import librosa
import pysrt
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import os
import soundfile

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

        if seq.ndim == 1:
            # mono
            audio_chunk = seq[ndx_start:ndx_end]
        else:
            # stereo (multi-channel)
            audio_chunk = seq[:, ndx_start:ndx_end]

        # NOTE: the subtitle may require some further processing
        # (e.g. to remove newlines, brackets, etc.)
        chunks.append((audio_chunk, sub.text))
    return chunks, sr


def store_chunks(audio_file: str, srt_file: str, csv_file: Optional[str]=None, outdir: str=".", out_extension: str="ogg") -> None:
    """Split an audio file based on its subtitles. Then, store the chunks in separate files
    and create a (csv) mapping file (audio file -> subtitle).

    Parameters
    ----------
    audio_file : str
        the path to the audio file (must be a format supported by librosa.load)
    srt_file : str
        the path to the subtitles file (must be a .srt file)
    csv_file : str
        the output csv file. If not specified, it is stored in `outdir` and has
        the same name as `audio_file` (minus the extension, which will be .csv)
    outdir : str
        the directory where the output files should be stored. `.` by default
    out_extension : str
        the extension of the output audio files. `ogg` by default. It should be
        an extension supported by `soundfile.write`.

    Returns
    -------
    None

    """
    basename = os.path.splitext(os.path.basename(audio_file))[0]

    if csv_file is None:
        csv_file = os.path.join(outdir, f"{basename}.csv")

    chunks, sr = extract_chunks(audio_file, srt_file)

    rows = []
    for i, (chunk, srt) in enumerate(chunks):
        outfile = os.path.join(outdir, f"{basename}.{i}.{out_extension}")
        soundfile.write(outfile, chunk, sr)
        rows.append((outfile, srt))

    df = pd.DataFrame(data=rows, columns=["audio", "subs"])
    df.to_csv(csv_file, index=False)


