import sys
import numpy as np
import librosa
sys.stderr.write('IGNORE ANY IMMEDIATELY FOLLOWING ERRORS. These are spuriously thrown by the pycochleagram module.\n')
from pycochleagram.cochleagram import cochleagram, invert_cochleagram
import soundfile

from .util import suppress_output, stderr

def wav_to_cochleagram(path, sr=16000, offset=10, low_lim=50, hi_lim=None, sample_factor=1, n_coef=13, order=2):
    assert sr % 1000 == 0, 'Must use a sampling rate that is a multiple of 1000'
    order = list(range(1, order+1))

    wav, _ = librosa.load(
        path,
        sr=sr
    )

    s = sample_factor
    n_in = (n_coef + 1 - 2*s) / s - 1
    assert int(n_in) > 0 and n_in % 1 == 0, 'The sample factor %d cannot be used to generate %d output filters. Use a choice of sample_factor s and n_coef for which (n + 1 - 2*s) / s - 1 is an integer greater than 0.'
    n_in = int(n_in)

    n_pcm = len(wav)
    downsample = int(1000 / offset)
    mod = int(sr / downsample)
    excess = (n_pcm) % mod
    if excess > 0:
        pad_len = mod - excess
    else:
        pad_len = 0
    if pad_len > 0:
        wav = np.pad(wav, ((0, pad_len),), 'constant')

    if hi_lim is None:
        hi_lim = min(int(sr/2), 20000)

    c = cochleagram(
        wav,
        sr,
        sample_factor=sample_factor,
        n=n_in, # Adds outer dims for high and low pass filters
        low_lim=low_lim,
        hi_lim=hi_lim,
        downsample=downsample
    )

    deltas = []

    for o in order:
        deltas.append(librosa.feature.delta(c, order=o))

    feats = np.concatenate([c] + deltas, axis=0)

    return feats, float(wav.shape[0]) / sr

def cochleagram_to_wav(cgram, fps=None, sr=16000, offset=10, low_lim=50, hi_lim=None, sample_factor=1, n_coef=13, n_iter=1):
    if fps is None:
        downsample = int(1000 / offset)
    else:
        downsample = fps
    if hi_lim is None:
        hi_lim = min(int(sr/2), 20000)

    s = sample_factor
    n_in = (n_coef + 1 - 2 * s) / s - 1
    assert int(n_in) > 0 and n_in % 1 == 0, 'The sample factor %d cannot be used to generate %d output filters. Use a choice of sample_factor s and n_coef for which (n + 1 - 2*s) / s - 1 is an integer greater than 0.'
    n_in = int(n_in)

    inverter = suppress_output(invert_cochleagram)

    wav, _ = inverter(cgram, sr, n_in, low_lim, hi_lim, sample_factor, downsample=downsample, n_iter=n_iter)

    return wav


def invert_cochleagrams(
        input=None,
        targets_bwd=None,
        preds_bwd=None,
        targets_fwd=None,
        preds_fwd=None,
        fps=None,
        sr=16000,
        offset=10,
        low_lim=50,
        hi_lim=None,
        sample_factor=1,
        n_iter=1,
        reverse_reconstructions=True,
        dir='./',
        prefix='',
        suffix='.aiff'
):
    fps = list(fps)
    if len(fps) == 1:
        fps *= 5
    cgram_types = []
    cgram_fps = []
    cgram_lists = []
    if input is not None:
        cgram_types.append('input')
        cgram_fps.append(fps[0])
        cgram_lists.append(input)
    if targets_bwd is not None:
        cgram_types.append('bwd_target')
        cgram_fps.append(fps[1])
        cgram_lists.append(targets_bwd)
    if preds_bwd is not None:
        cgram_types.append('bwd_pred')
        cgram_fps.append(fps[2])
        cgram_lists.append(preds_bwd)
    if targets_fwd is not None:
        cgram_types.append('fwd_target')
        cgram_fps.append(fps[3])
        cgram_lists.append(targets_fwd)
    if preds_fwd is not None:
        cgram_types.append('fwd_pred')
        cgram_fps.append(fps[4])
        cgram_lists.append(preds_fwd)

    if len(cgram_types) > 0:
        for i in range(len(cgram_lists[0])):
            for j in range(len(cgram_types)):
                name = cgram_types[j]
                cgram_cur = cgram_lists[j][i]
                fps_cur = cgram_fps[j]
                cgram_cur = np.swapaxes(cgram_cur, -2, -1)
                n_coef = cgram_cur.shape[0]
                path = dir + '/' + prefix + '%d_' % i + name + suffix

                if reverse_reconstructions and name.startswith('bwd'):
                    cgram_cur = cgram_cur[:,::-1]

                wav = cochleagram_to_wav(
                    cgram_cur,
                    fps=fps_cur,
                    sr=sr,
                    offset=offset,
                    low_lim=low_lim,
                    hi_lim=hi_lim,
                    sample_factor=sample_factor,
                    n_coef=n_coef,
                    n_iter=n_iter
                )

                try:
                    soundfile.write(path, wav, sr, format='AIFF')
                except RuntimeError:
                    stderr('Error saving audio to path: %s. Skipping...\n' % path)
