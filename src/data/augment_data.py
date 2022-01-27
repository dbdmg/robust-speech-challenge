import torch
from torch_audiomentations import Compose, AddBackgroundNoise, HighPassFilter, LowPassFilter, PitchShift

def augment_samples (
    list_samples: list,
    noise_root_path: str,
    sample_rate: int = 16_000,
    signal_key: str = "array",
    device: str = "cpu",
    ):

    apply_augmentation = Compose(
        transforms=[
            AddBackgroundNoise(
                background_paths=noise_root_path,
                p=0.5,
            ),
            HighPassFilter(
                p=0.25
            ),
            LowPassFilter(
                p=0.25
            ),
            PitchShift(
                sample_rate = sr,
                min_transpose_semitones = -2.0,
                max_transpose_semitones = 2.0,
                p=0.05
            )
        ]
    )

    audio_samples = [d[signal_key] for d in list_samples]
    augmented_signals = pply_augmentation(audio_samples, sample_rate=sample_rate)

    for i, d in list_samples:
        list_samples[i][signal_key] = augmented_signals[i]

    return list_samples


