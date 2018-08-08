import numpy as np
import librosa

def wav_to_mfcc(path, sr=16000, offset=10, window_len=25, n_coef=13, order=2):
    assert sr % 1000 == 0, 'Must use a sampling rate that is a multiple of 1000'
    order = list(range(1, order+1))

    pts_per_ms = int(sr / 1000)

    wav, _ = librosa.load(
        path,
        sr=sr
    )

    mfcc = librosa.feature.mfcc(
        y=wav,
        sr=sr,
        n_mfcc=n_coef,
        n_fft=pts_per_ms * window_len,
        hop_length=pts_per_ms * offset
    )

    deltas = []

    for o in order:
        deltas.append(librosa.feature.delta(mfcc, order=o))

    feats = np.concatenate([mfcc] + deltas, axis=0)

    return feats
