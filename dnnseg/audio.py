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

    return feats, float(wav.shape[0]) / sr

def wav_to_cochleagram(path, sr=16000, offset=10, sample_factor=2, n_coef=13, order=2):
    from pycochleagram.cochleagram import human_cochleagram, apply_envelope_downsample
    assert sr % 1000 == 0, 'Must use a sampling rate that is a multiple of 1000'
    order = list(range(1, order+1))

    pts_per_ms = int(sr / 1000)

    wav, _ = librosa.load(
        path,
        sr=sr
    )

    print(wav.shape)

    hi_lim = min(int(sr/2), 20000)

    cochleagram = human_cochleagram(
        wav,
        sr,
        n=n_coef,
        hi_lim=hi_lim
    )

    print('Cochleagram computed')

    cochleagram = apply_envelope_downsample(
        cochleagram,
        'resample',
        audio_sr=sr,
        sample_factor=sample_factor,
        env_sr=int(sr/offset)
    )

    print('Cochleagram downsampled')

    print(cochleagram.shape)

    deltas = []

    for o in order:
        deltas.append(librosa.feature.delta(cochleagram, order=o))

    feats = np.concatenate([cochleagram] + deltas, axis=0)

    return feats, float(wav.shape[0]) / sr