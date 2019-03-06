import sys
import os
import math
import time
import numpy as np
import scipy.signal
import pandas as pd
from scipy.interpolate import Rbf
from .audio import wav_to_mfcc


def binary_to_integer_np(b, int_type='int32'):
    np_int_type = getattr(np, int_type)

    k = int(b.shape[-1])
    if int_type.endswith('128'):
        assert k <= int(int_type[-3:]) / 2, 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' %(b.shape[-1], int_type)
    else:
        assert k <= int(int_type[-2:]) / 2, 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' %(b.shape[-1], int_type)
    base2 = 2 ** np.arange(k-1, -1, -1, dtype=np_int_type)
    while len(base2.shape) < len(b.shape):
        base2 = np.expand_dims(base2, 0)

    return (b.astype(np_int_type) * base2).sum(axis=-1)


def binary_to_string_np(b):
    b_str = np.char.mod('%d', b.astype('int'))
    string = list(map(''.join, b_str))

    return string


def get_segmentation_smoother(times, segs, smoothing_factor=1., function='multiquadric'):
    out = Rbf(times, segs, smooth=smoothing_factor, function=function)

    return out


def extract_peak_ix(y):
    before = np.full_like(y, np.inf)
    before[..., 1:] = y[..., :-1]
    after = np.full_like(y, np.inf)
    after[..., :-1] = y[..., 1:]

    ix = np.where(np.logical_and(y > before, y > after))

    return ix


def smooth_segmentations(segs, smoothing_factor=1., function='multiquadric', seconds_per_step=0.01, n_points=None):
    n_frames = len(segs)
    max_len = n_frames * seconds_per_step
    x = np.linspace(0, max_len, n_frames)
    y = segs
    if n_points is None:
        n_points = n_frames
    basis = np.linspace(0, max_len, n_points)

    out = get_segmentation_smoother(x, y, smoothing_factor=smoothing_factor, function=function)(basis)

    return basis, out


def extract_segment_timestamps(
        segs,
        algorithm=None,
        algorithm_params=None,
        seconds_per_step=0.01,
        n_points=None,
        return_plot=False
):
    assert not (algorithm is None and n_points), 'Extraction at custom timepoints (n_points != None) is not supported when smoothing is turned of (algorithm==None).'
    n_frames = len(segs)
    max_len = n_frames * seconds_per_step
    x = np.linspace(0, max_len, n_frames)
    y = segs
    if algorithm_params is None:
        algorithm_params = {}

    if algorithm == 'rbf':
        if n_frames > 1:
            smoother = get_segmentation_smoother(x, y, **algorithm_params)

            if n_points is None:
                n_points = n_frames
            basis = np.linspace(0, max_len, n_points)
            response = smoother(basis)

            seg_ix = extract_peak_ix(response)[0]
            timestamps = basis[seg_ix]
        elif n_frames == 1 and x[0] > 0:
            timestamps = x
        else:
            timestamps = np.array([max_len])

        if len(timestamps) == 0 or timestamps[-1] < max_len - 0.02:
            timestamps = np.concatenate([timestamps, [max_len]], axis=0)

    elif algorithm is None:
        if n_points is None:
            n_points = n_frames
        seg_ix = np.where(segs)[0]
        basis = np.linspace(0, max_len, n_points)
        response = segs
        if len(seg_ix) > 0:
            timestamps = seg_ix * seconds_per_step + seconds_per_step / 2
        else:
            timestamps = np.array([max_len])

    else:
        raise ValueError('Smoothing algorithm %s not supported at this time' %algorithm)

    if return_plot:
        out = (timestamps, basis, response)
    else:
        out = timestamps


    return out


def extract_segment_timestamps_batch(
        segs,
        algorithm=None,
        algorithm_params=None,
        seconds_per_step=0.01,
        n_points=None,
        return_plot=False,
        mask=None,
        padding=None
):
    timestamps = []
    if return_plot:
        basis = np.zeros((len(segs), n_points))
        response = np.zeros((len(segs), n_points))

    if mask is not None:
        sequence_lengths = np.sum(mask, axis=1).astype('int')

    for i in range(len(segs)):
        segs_i = segs[i]
        if mask is not None:
            if padding == 'pre':
                segs_i = segs_i[-sequence_lengths[i]:]
            elif padding == 'post':
                segs_i = segs_i[:-sequence_lengths[i]]

        extraction = extract_segment_timestamps(
            segs_i,
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            seconds_per_step=seconds_per_step,
            n_points=n_points,
            return_plot=return_plot
        )
        if return_plot:
            timestamps_cur, basis_cur, response_cur = extraction
        else:
            timestamps_cur = extraction

        timestamps.append(timestamps_cur)

        if return_plot:
            basis[i] = basis_cur
            response[i] = response_cur

    if return_plot:
        out = timestamps, basis, response
    else:
        out = timestamps

    return out


def extract_states_at_timestamps(
        timestamps,
        states,
        steps_per_second=100.,
        activation='tanh',
        discretize=True,
        as_categories=False,
        as_onehot=True
):
    assert activation in ['sigmoid', 'softmax', 'tanh', 'linear', None]
    assert not (activation in ['linear', None] and discretize), "States with linear activation can't be discretized."
    assert not as_categories and as_onehot, 'Labels cannot be both categorical (integer) and one-hot.'

    ix = np.minimum(np.floor(timestamps * steps_per_second), len(states) - 1).astype('int')
    out = states[ix]

    if discretize:
        if activation in ['sigmoid', 'tanh']:
            if activation == 'tanh':
                out = (out + 1) / 2
            out = np.rint(out).astype('int')

            if as_categories:
                out = binary_to_integer_np(out)
            elif as_onehot:
                out = binary_to_string_np(out)

        elif activation == 'softmax':
            if as_categories:
                out = np.argmax(out, axis=-1)
            else:
                one_hot = np.zeros_like(out)
                one_hot[np.argmax(out, axis=-1)[..., None]] = 1
                out = one_hot.astype('int')

    return out


def extract_states_at_timestamps_batch(
        timestamps,
        states,
        steps_per_second=100.,
        activation='tanh',
        discretize=True,
        as_categories=True,
        as_onehot=True,
        mask=None,
        padding=None
):
    out = []

    if mask is not None:
        sequence_lengths = np.sum(mask, axis=1).astype('int')

    for i in range(len(states)):
        states_i = states[i]
        if mask is not None:
            if padding == 'pre':
                states_i = states_i[-sequence_lengths[i]:]
            elif padding == 'post':
                states_i = states_i[:-sequence_lengths[i]]

        out_cur = extract_states_at_timestamps(
            timestamps[i],
            states_i,
            steps_per_second=steps_per_second,
            activation=activation,
            discretize=discretize,
            as_categories=as_categories,
            as_onehot=as_onehot
        )

        out.append(out_cur)

    return out


def binary_segments_to_intervals_inner(binary_segments, mask, src_segments=None, labels=None, seconds_per_step=0.01):
    assert binary_segments.shape == mask.shape, 'binary_segments and mask must have the same shape'

    seg_indices1, seg_indices2 = np.where(binary_segments * mask)

    if src_segments is None:
        timestamps_ends = (np.reshape(mask.cumsum(), mask.shape)) * seconds_per_step
        ends = timestamps_ends[seg_indices1, seg_indices2]
        out = pd.DataFrame({'end': ends})
        out['start'] = out.end.shift().fillna(0.)
    else:
        timestamps_ends = mask.cumsum(axis=1) * seconds_per_step
        assert len(src_segments) == len(timestamps_ends), 'Length mismatch between inputs (%d) and input segment boundaries (%d)' % (len(timestamps_ends), len(src_segments))

        src_segments_starts = np.array(src_segments.start)[seg_indices1]
        src_segments_ends = np.array(src_segments.end)[seg_indices1]
        src_segments_indices = np.arange(len(src_segments))[seg_indices1]

        ends = timestamps_ends[seg_indices1, seg_indices2]
        out = pd.DataFrame({'end': ends})

        out['start'] = out.end.groupby(src_segments_indices).shift().fillna(0.)

        out.start += src_segments_starts
        out.end += src_segments_starts
        out.end = np.minimum(out.end, src_segments_ends)

    if labels is None:
        out['label'] = 0
    else:
        out['label'] = labels[np.where(binary_segments * mask)]
    out['index'] = np.arange(len(out))

    out.start = out.start.round(decimals=3)
    out.end = out.end.round(decimals=3)

    return out


def binary_segments_to_intervals(binary_segments, mask, file_indices, src_segments=None, labels=None, seconds_per_step=0.01):
    fileIDs = sorted(list(file_indices.keys()))
    out = []
    for f in fileIDs:
        s, e = file_indices[f]
        file_binary_segments = binary_segments[s:e]
        file_mask = mask[s:e]
        if src_segments is None:
            file_src_segments = src_segments
        else:
            file_src_segments = src_segments.iloc[s:e]

        file_intervals = binary_segments_to_intervals_inner(
            file_binary_segments,
            file_mask,
            src_segments=file_src_segments,
            labels=labels,
            seconds_per_step=seconds_per_step
        )
        file_intervals['fileID'] = f
        out.append(file_intervals)

    return pd.concat(out, axis=0)


def pad_sequence(sequence, seq_shape=None, dtype='float32', reverse=False, padding='pre', value=0.):
    assert padding in ['pre', 'post'], 'Padding type "%s" not recognized' % padding
    if seq_shape is None:
        seq_shape = get_seq_shape(sequence)

    if len(seq_shape) == 0:
        return sequence
    if isinstance(sequence, list):
        sequence = np.array([pad_sequence(x, seq_shape=seq_shape[1:], dtype=dtype, reverse=reverse, padding=padding, value=value) for x in sequence])
        pad_width = seq_shape[0] - len(sequence)
        if padding == 'pre':
            pad_width = (pad_width, 0)
        elif padding == 'post':
            pad_width = (0, pad_width)
        if len(seq_shape) > 1:
            pad_width = [pad_width]
            pad_width += [(0, 0)] * (len(seq_shape) - 1)
        sequence = np.pad(
            sequence,
            pad_width=pad_width,
            mode='constant',
            constant_values=(value,)
        )
    elif isinstance(sequence, np.ndarray):
        pad_width = [(seq_shape[i] - sequence.shape[i], 0) if padding=='pre' else (0, seq_shape[i] - sequence.shape[i]) for i in range(len(sequence.shape))]

        if reverse:
            sequence = sequence[::-1]

        sequence = np.pad(
            sequence,
            pad_width=pad_width,
            mode='constant',
            constant_values=(value,)
        )

    return sequence


def repad_acoustic_features(acoustic_features, new_length, padding='pre', value=0.):
    if new_length >= acoustic_features.shape[-2]:
        new_feats = np.full(acoustic_features.shape[:-2] + (new_length,) + acoustic_features.shape[-1:], fill_value=value)
        old_length = acoustic_features.shape[-2]
        if padding == 'pre':
            new_feats[:,-old_length:,:] = acoustic_features
        else:
            new_feats[:,:old_length,:] = acoustic_features
    else:
        if padding=='pre':
            new_feats = acoustic_features[:,-new_length:,:]
        else:
            new_feats = acoustic_features[:,:new_length,:]

    return new_feats


def get_seq_shape(seq):
    seq_shape = []
    if isinstance(seq, list):
        seq_shape = [len(seq)]
        child_seq_shapes = [get_seq_shape(x) for x in seq]
        for i in range(len(child_seq_shapes[0])):
            seq_shape.append(max([x[i] for x in child_seq_shapes]))
    elif isinstance(seq, np.ndarray):
        seq_shape += seq.shape

    return seq_shape


def get_random_permutation(n):
    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv


def segment_length_summary(segs, indent=0, steps_per_second=100.):
    seg_min = (segs.end - segs.start).min()
    seg_max = (segs.end - segs.start).max()
    seg_95 = (segs.end - segs.start).quantile(q=0.95)
    seg_99 = (segs.end - segs.start).quantile(q=0.99)
    seg_999 = (segs.end - segs.start).quantile(q=0.999)
    seg_mean = (segs.end - segs.start).mean()
    out = ' ' * indent + 'Segment length statistics:\n'
    out += ' ' * (indent + 2) + 'Seconds:\n'
    out += ' ' * (indent + 4) + 'Min: %.4f\n' % (seg_min)
    out += ' ' * (indent + 4) + 'Max: %.4f\n' % (seg_max)
    out += ' ' * (indent + 4) + '95th percentile: %.4f\n' % (seg_95)
    out += ' ' * (indent + 4) + '99th percentile: %.4f\n' % (seg_99)
    out += ' ' * (indent + 4) + '99.9th percentile: %.4f\n' % (seg_999)
    out += ' ' * (indent + 4) + 'Mean: %.4f\n' % (seg_mean)

    if steps_per_second:
        out += ' ' * (indent + 2) + 'Frames:\n'
        out += ' ' * (indent + 4) + 'Min: %.4f\n' % (seg_min * steps_per_second)
        out += ' ' * (indent + 4) + 'Max: %.4f\n' % (seg_max * steps_per_second)
        out += ' ' * (indent + 4) + 'Max: %.4f\n' % (seg_max * steps_per_second)
        out += ' ' * (indent + 4) + '95th percentile: %.4f\n' % (seg_95 * steps_per_second)
        out += ' ' * (indent + 4) + '99th percentile: %.4f\n' % (seg_99 * steps_per_second)
        out += ' ' * (indent + 4) + '99.9th percentile: %.4f\n' % (seg_999 * steps_per_second)
        out += ' ' * (indent + 4) + 'Mean: %.4f\n' % (seg_mean * steps_per_second)

    return out


def score_segmentation(true, pred, tol=0.02):
    true = np.array(true[['start','end']])
    pred = np.array(pred[['start','end']])

    i = 0
    j = 0

    b_tp = 0
    b_fp = 0
    b_fn = 0

    w_tp = 0
    w_fp = 0
    w_fn = 0

    e_true_prev = true[0, 0]
    e_pred_prev = pred[0, 0]

    while i < len(true) or j < len(pred):
        if i >= len(true):
            # All gold segments have been read.
            # Scan to the end of the predicted segments and tally up precision penalties.

            s_pred, e_pred = pred[j]

            jump_pred = s_pred - e_pred_prev > 1e-5
            if jump_pred:
                e_pred = s_pred

            j += not jump_pred
            b_fp += 1
            w_fp += not jump_pred

            e_pred_prev = e_pred

        elif j >= len(pred):
            # All predicted segments have been read.
            # Scan to the end of the true segments and tally up recall penalties.

            s_true, e_true = true[i]

            jump_true = s_true - e_true_prev > 1e-5
            if jump_true:
                e_true = s_true

            i += not jump_true
            b_fn += 1
            w_fn += not jump_true

            e_true_prev = e_true

        else:
            # Neither true nor pred have finished

            s_true, e_true = true[i]
            s_pred, e_pred = pred[j]

            # If there is a "jump" in the true segs, create a pseudo segment spanning the jump
            jump_true = s_true - e_true_prev > 1e-5
            if jump_true:
                e_true = s_true
                s_true = e_true_prev

            # If there is a "jump" in the predicted segs, create a pseudo segment spanning the jump
            jump_pred = s_pred - e_pred_prev > 1e-5
            if jump_pred:
                e_pred = s_pred
                s_pred = e_pred_prev

            # Compute whether starts and ends of true and predicted segments align within the tolerance
            s_hit = False
            e_hit = False

            s_diff = math.fabs(s_true-s_pred)
            e_diff = math.fabs(e_true-e_pred)

            if s_diff <= tol:
                s_hit = True
            if e_diff <= tol:
                e_hit = True

            if s_hit:
                # Starts align, move on to the next segment in both feeds.
                # If we are in a pseudo-segment, don't move the pointer,
                # just update ``e_true_prev`` and ``e_true_prev``.

                i += not jump_true
                j += not jump_pred

                b_tp += 1

                # Only update word score tallies for non-pseudo-segments
                w_tp += e_hit and not (jump_true or jump_pred)
                w_fp += not e_hit and not jump_pred
                w_fn += not e_hit and not jump_true

                e_true_prev = e_true
                e_pred_prev = e_pred

            elif s_true < s_pred:
                # Starts did not align and the true segment starts before the predicted one.
                # Move on to the next segment in the true feed and tally recall penalties.
                # If we are in a pseudo-segment, don't move the pointer,
                # just update ``e_true_prev``.
                i += not jump_true

                b_fn += 1
                # Only update word score tallies for non-pseudo-segments
                w_fn += not jump_true

                e_true_prev = e_true

            else:
                # Starts did not align and the predicted segment starts before the true one.
                # Move on to the next segment in the predicted feed and tally precision penalties.
                # If we are in a pseudo-segment, don't move the pointer,
                # just update ``e_pred_prev``.
                j += not jump_pred

                b_fp += 1
                # Only update word score tallies for non-pseudo-segments
                w_fp += not jump_pred

                e_pred_prev = e_pred

    # Score final boundary
    hit = math.fabs(e_true_prev - e_pred_prev) <= tol
    b_tp += hit
    b_fp += not hit
    b_fn += not hit

    out = {
        'b_tp': b_tp,
        'b_fp': b_fp,
        'b_fn': b_fn,
        'w_tp': w_tp,
        'w_fp': w_fp,
        'w_fn': w_fn
    }

    return out


def precision_recall(n_matched, n_true, n_pred):
    """Calculates the classification precision and recall, given
    the number of true positives, the number of existing positives,
    and the number of proposed positives."""

    if n_true > 0:
        recall = n_matched / n_true
    else:
        recall = 0.0

    if n_pred > 0:
        precision = n_matched / n_pred
    else:
        precision = 0.0

    return precision, recall


def fscore(precision, recall, beta=1.0):
    """Calculates the f-score (default is balanced f-score; beta > 1
    favors precision), the harmonic mean of precision and recall."""

    num = (beta**2 + 1) * precision * recall
    denom = (beta**2) * precision + recall
    if denom == 0:
        return 0.0
    else:
        return num / denom


def precision_recall_f(n_matched, n_true, n_pred, beta=1.0):
    """Calculates precision, recall and f-score."""

    prec,rec = precision_recall(n_matched, n_true, n_pred)
    f = fscore(prec, rec, beta=beta)

    return prec,rec,f


def get_text_boundaries(words):
    breaks = []
    soFar = 0
    # Don't count the first break (utterance beginning)
    for word in words:
        soFar += len(word)
        breaks.append(soFar)
    # Don't count the last break (utterance end)
    breaks.pop(-1)
    return breaks


def get_text_word_starts_and_ends(words):
    bounds = []

    x0 = 0
    for ii, wi in enumerate(words):
        (b1,b2) = (x0, x0 + len(words[ii]))
        bounds.append((b1, b2))
        x0 += len(words[ii])

    return bounds


def score_text_words(true, pred):
    tp = 0
    fp = 0
    fn = 0

    for w_true, w_pred in zip(true, pred):
        bounds_true = set(get_text_word_starts_and_ends(w_true))
        bounds_pred = set(get_text_word_starts_and_ends(w_pred))

        tp += len(bounds_true.intersection(bounds_pred))
        fp += len(bounds_pred - bounds_true)
        fn += len(bounds_true - bounds_pred)

    return tp, fp, fn


def score_text_lexicon(true, pred):
    lex_true = set()
    lex_pred = set()

    for w_true, w_pred in zip(true, pred):
        for word in w_true:
            if type(word) == list:
                word = tuple(word)
            lex_true.add(word)

        for word in w_pred:
            if type(word) == list:
                word = tuple(word)
            lex_pred.add(word)

    tp = len(lex_true.intersection(lex_pred))
    fp = len(lex_pred - lex_true)
    fn = len(lex_true - lex_pred)

    return tp, fp, fn


def score_text_boundaries(true, pred):
    tp = 0
    fp = 0
    fn = 0

    warned = 0

    for w_true, w_pred in zip(true, pred):
        # print "true:", w_true
        # print "pred:", w_pred

        if type(w_true[0]) == str:
            cat1 = "".join(w_true)
            cat2 = "".join(w_pred)
        else:
            cat1 = sum(w_true, [])
            cat2 = sum(w_pred, [])

        if cat1 != cat2:
            if not warned:
                print("Warning: surface string mismatch:", cat1, cat2)
                warned = 1
            elif warned == 1:
                print("Warning: more mismatches")
                warned += 1

        boundaries_true = set(get_text_boundaries(w_true))
        boundaries_pred = set(get_text_boundaries(w_pred))

        tp += len(boundaries_true.intersection(boundaries_pred))
        fp += len(boundaries_pred - boundaries_true)
        fn += len(boundaries_true - boundaries_pred)

    return tp, fp, fn


def text_to_wordlists(text):
    out = []
    for l in text.splitlines():
        out.append(l.split())
    return out


def score_text_segmentation(true, pred):
    p = text_to_wordlists(pred)
    t = text_to_wordlists(true)

    b_tp, b_fp, b_fn = score_text_boundaries(t, p)
    w_tp, w_fp, w_fn = score_text_words(t, p)
    l_tp, l_fp, l_fn = score_text_lexicon(t, p)

    out = {
        'b_tp': b_tp,
        'b_fp': b_fp,
        'b_fn': b_fn,
        'w_tp': w_tp,
        'w_fp': w_fp,
        'w_fn': w_fn,
        'l_tp': l_tp,
        'l_fp': l_fp,
        'l_fn': l_fn,
    }

    return out


def compute_unsupervised_classification_targets(true, pred):
    true_src = true
    true = np.array(true[['start','end','label']])
    pred = np.array(pred[['start','end','label']])

    i = 0
    j = 0

    true_labels = []
    candidates = []

    while i < len(true) and j < len(pred):

        s_t, e_t, l_t = true[i]
        s_p, e_p, l_p = pred[j]

        inside_l = s_t >= s_p and s_t < e_p
        inside_r = e_t > s_p and e_t <= e_p
        inside = inside_l or inside_r

        if inside:
            overlap = min(e_t, e_p) - max(s_t, s_p)
            candidates.append((overlap, l_t))
            i += 1
        else:
            assert len(candidates) > 0, 'Position %d has no label candidates and no overlapping true segments' % j
            winner = max(candidates, key = lambda x: x[0])[1]
            true_labels.append(winner)
            candidates = []
            j += 1

    assert len(candidates) > 0, 'Position %d has no label candidates and no overlapping true segments' % j
    winner = max(candidates, key=lambda x: x[0])[1]
    true_labels.append(winner)

    out = true_src.copy()
    out.label = true_labels

    return out


def get_padded_lags(X, n, backward=True, clip_feat=None):
    # X is an array with shape [batch, time, feat] or list of such arrays
    out = []
    not_list = False
    if not isinstance(X, list):
        not_list = True
        X = [X]
    for x in X:
        shape = x.shape
        new_shape = list(shape[0:2]) + [n] + list(shape[2:])

        out_cur = np.zeros(new_shape)
        if clip_feat:
            max_feat = clip_feat
        else:
            max_feat = x.shape[-1]

        for i in range(n):
            end = x.shape[1] - i
            if backward:
                out_cur[:, i, i:, :max_feat] = x[:, :end, :max_feat]
            else:
                out_cur[:, i, :end, :max_feat] = x[:, i:, :max_feat]

        out.append(out_cur)

    if not_list:
        out = out[0]

    return out


class Dataset(object):
    def __new__(cls, *args, **kwargs):
        if cls is Datafile:
            raise TypeError("Dataset is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(
            self,
            dir_path,
            datatype='acoustic',
            clip_timesteps=None,
            verbose=True,
            **kwargs
    ):

        self.dir_path = dir_path
        self.datatype = datatype.lower()
        assert self.datatype in ['acoustic', 'text'], 'Unrecognized datatype requested: "%s"' % self.datatype
        self.clip_timesteps = clip_timesteps

        self.data = {}
        self.cache = {}

        data_kwargs = {}
        if self.datatype == 'acoustic':
            suffix = '.wav'
            datafile = AcousticDatafile
            defaults = {
                'sr': 16000,
                'offset': 10,
                'window_len': 25,
                'n_coef': 13,
                'order': 2
            }
            for x in ['sr', 'offset', 'window_len', 'n_coef', 'order']:
                if x in kwargs:
                    data_kwargs[x] = kwargs[x]
                    setattr(self, x, kwargs[x])
                else:
                    setattr(self, x, defaults[x])
        else:
            suffix = '.txt'
            datafile = TextDatafile
            if 'lower' in kwargs:
                data_kwargs['lower'] = kwargs['lower']
                setattr(self, 'lower', kwargs['lower'])
            else:
                setattr(self, 'lower', False)

        files = ['%s/%s' % (self.dir_path, x) for x in os.listdir(self.dir_path) if x.lower().endswith(suffix)]
        n = len(files)

        times = []
        for i, f in enumerate(files):
            t0 = time.time()
            if verbose:
                file_name = f.split('/')[-1]
                if len(file_name) > 10:
                    file_name = file_name[:7] + '...'
                out_str = '\rProcessing file "%s" %d/%d' % (file_name, i + 1, n)
                out_str += ' ' * (40 - len(out_str))
                if i > 0:
                    h = int(eta / 3600)
                    r = int(eta % 3600)
                    m = int(r / 60)
                    s = int(r % 60)
                    if h > 0:
                        time_str = str(h) + ':%02d' % m + ':%02d' % s
                    else:
                        if m > 0:
                            time_str = str(m) + ':' + '%02d' % s
                        else:
                            time_str = '%ds' % s
                    out_str += '|    ETA - %s     ' % time_str
                sys.stderr.write(out_str)

            new_data = datafile(
                f,
                clip_timesteps=clip_timesteps,
                **data_kwargs
            )

            self.data[new_data.ID] = new_data
            t1 = time.time()
            times.append(t1 - t0)
            mean_time = sum(times) / len(times)
            eta = (n - i + 1) * mean_time

        if verbose:
            sys.stderr.write('\n')

        self.fileIDs = sorted(list(self.data.keys()))

        if self.datatype == 'text':
            ix2char = set()
            for f in self.fileIDs:
                ix2char = ix2char.union(set(self.data[f].ix2char))
            self.ix2char = sorted(list(ix2char))
            self.char2ix = {}
            for i, x in enumerate(self.ix2char):
                self.char2ix[x] = i
            assert not ' ' in self.ix2char, 'Space " " is a reserved character and cannot be used in text inputs to DNNSeg.'
            self.char2ix[' '] = len(self.ix2char)
            self.ix2char.append(' ')
            for f in self.fileIDs:
                self.data[f].update_charset(self.ix2char, self.char2ix)

        self.vad_perm = None
        self.vad_perm_inv = None
        self.wrd_perm = None
        self.wrd_perm_inv = None
        self.phn_perm = None
        self.phn_perm_inv = None

        if self.datatype == 'acoustic':
            self.steps_per_second = 1000. / self.offset
            self.seconds_per_step = float(self.offset) / 1000.
        else:
            self.offset = 1.
            self.steps_per_second = 1.
            self.seconds_per_step = 1.

    def features(
            self,
            features=None,
            boundaries_as_features=False,
            fold=None,
            mask=None,
            pad_left=None,
            pad_right=None,
            pad_final_one=False,
            normalize=False,
            center=False
    ):
        out = []
        boundaries = []
        new_series = []

        if pad_left is None:
            pad_left = 0
        if pad_right is None:
            pad_right = 0

        for f in self.fileIDs:
            if mask:
                feats, _ = self.data[f].segment_and_stack(
                    segments=mask,
                    features=features,
                    boundaries_as_features=boundaries_as_features,
                    padding=None,
                    pad_final_one=pad_final_one,
                )

                boundaries_file = []
                for x in feats:
                    boundaries_cur = np.zeros(x.shape[:-1])
                    boundaries_cur[..., -1] = 1
                    boundaries_file.append(boundaries_cur)

                feats = np.concatenate(
                    feats,
                    axis=1
                )
                boundaries_file = np.concatenate(boundaries_file, axis=1)

            else:
                feats = self.data[f].data()[None, ...]
                boundaries_file = np.zeros(feats.shape[:-1])

            if normalize:
                maximum = feats.max()
                minimum = feats.min()
                diff = maximum - minimum
                feats = (feats - minimum) / diff
            if center:
                feats = feats - feats.mean()

            if pad_left or pad_right:
                pad_width = []
                for i in range(len(feats.shape)):
                    if i == len(feats.shape) - 2:
                        pad_width.append((pad_left, pad_right))
                    else:
                        pad_width.append((0, 0))
                feats = np.pad(feats, pad_width, mode='constant')
                boundaries_file = np.pad(boundaries_file, pad_width[:-1], mode='constant')

            if fold:
                to_add_feats = []
                to_add_boundaries = []
                for i in range(0, feats.shape[1], fold):
                    if i == 0:
                        new_series.append(1)
                    else:
                        new_series.append(0)
                    to_add_feats.append(feats[:, i:i+fold])
                    to_add_boundaries.append(boundaries_file[:, i:i+fold])
                out += to_add_feats
                boundaries += to_add_boundaries
            else:
                new_series.append(1)
                out.append(feats)
                boundaries.append(boundaries_file)

        new_series = np.array(new_series)

        return out, boundaries, new_series

    def segment_and_stack(
            self,
            features=None,
            boundaries_as_features=False,
            segments='vad',
            max_len=None,
            padding='pre',
            pad_final_one=False,
            reverse=False,
            normalize=False,
            center=False,
            with_deltas=True,
            resample=None
    ):
        feats = []
        mask = []
        pad_seqs = padding not in ['None', None]

        for f in self.fileIDs:
            if isinstance(segments, dict):
                segments_cur = segments[f]
            else:
                segments_cur = segments
            if len(segments_cur) > 0:
                new_feats, new_mask = self.data[f].segment_and_stack(
                    features=features[f],
                    boundaries_as_features=boundaries_as_features,
                    segments=segments_cur,
                    max_len=max_len,
                    padding=padding,
                    pad_final_one=pad_final_one,
                    reverse=reverse,
                    normalize=normalize,
                    center=center,
                    with_deltas=with_deltas,
                    resample = resample,
                )
                if pad_seqs:
                    feats.append(new_feats)
                    mask.append(new_mask)
                else:
                    feats += new_feats
                    mask += new_mask

        if pad_seqs:
            if padding not in ['None', None]:
                max_len = 0
                for f in feats:
                    max_len = max(max_len, f.shape[-2])
                for i, f in enumerate(feats):
                    feats[i] = repad_acoustic_features(feats[i], max_len, padding=padding)
                for i, f in enumerate(mask):
                    mask[i] = np.squeeze(repad_acoustic_features(mask[i][...,None], max_len, padding=padding), -1)

            feats = np.concatenate(feats, axis=0)
            mask = np.concatenate(mask, axis=0)

        return feats, mask

    def cache_utterance_data(
            self,
            name,
            segments='vad',
            max_len=None,
            normalize_inputs=False,
            center_inputs=False,
            normalize_targets=False,
            center_targets=False,
            input_padding='pre',
            input_resampling=None,
            target_padding='post',
            target_resampling=None,
            reverse_targets=True,
            predict_deltas=False,
            forced_boundaries=None
    ):
        X, X_mask = self.inputs(
            segments=segments,
            padding=input_padding,
            max_len=max_len,
            normalize=normalize_inputs,
            center=center_inputs,
            resample=input_resampling
        )
        y, y_mask = self.targets(
            segments=segments,
            padding=target_padding,
            max_len=max_len,
            reverse=reverse_targets,
            normalize=normalize_targets,
            center=center_targets,
            with_deltas=predict_deltas,
            resample=target_resampling
        )
        speaker = self.segments(segments).speaker.values

        if forced_boundaries:
            inner_segment_type = forced_boundaries
            gold_boundaries, _ = self.one_hot_boundaries(
                inner_segments=inner_segment_type,
                outer_segments=segments,
                padding=input_padding,
                max_len=max_len
            )
        else:
            gold_boundaries = None

        cache_dict = {
            'type': 'utterance',
            'n': len(y),
            'X': X,
            'X_mask': X_mask,
            'y': y,
            'y_mask': y_mask,
            'speaker': speaker,
            'gold_boundaries': gold_boundaries
        }

        self.cache[name] = cache_dict

    def cache_streaming_data(
            self,
            name,
            window_len_input,
            window_len_bwd,
            window_len_fwd,
            normalize_inputs=False,
            center_inputs=False,
            normalize_targets=False,
            center_targets=False,
            target_bwd_resampling=None,
            target_fwd_resampling = None,
            predict_deltas=False,
            forced_boundaries=None,
            mask=None
    ):
        left_pad = max(window_len_input, window_len_bwd) - 1
        right_pad = window_len_fwd

        feats_tmp, boundaries_tmp, _ = self.features(
            mask=mask,
            pad_left=left_pad,
            pad_right=right_pad,
            normalize=normalize_inputs,
            center=center_inputs
        )
        feats_inputs = []
        boundaries = []
        file_lengths = []
        for i, x in enumerate(feats_tmp):
            file_lengths.append(len(x[0]) - (left_pad + right_pad))
            feats_inputs.append(x[0])
            boundaries.append(boundaries_tmp[i][0])

        n = sum(file_lengths)
        feats_inputs = pad_sequence(feats_inputs, padding='post')
        boundaries = pad_sequence(boundaries, padding='post')

        if (normalize_inputs == normalize_targets) and (center_inputs == center_targets):
            feats_targets = feats_inputs
        else:
            feats_tmp, _, _ = self.features(
                mask=mask,
                pad_left=left_pad,
                pad_right=right_pad,
                normalize=normalize_targets,
                center=center_targets
            )
            feats_targets = []
            file_lengths = []
            for i, x in enumerate(feats_tmp):
                file_lengths.append(len(x[0]) - (left_pad + right_pad))
                feats_targets.append(x[0])

            feats_targets = pad_sequence(feats_targets, padding='post')

        file_ix = []
        time_ix = []

        for i, l in enumerate(file_lengths):
            file_ix_cur = np.ones(l, dtype='int') * i
            time_ix_cur = np.arange(left_pad, left_pad + l)

            file_ix.append(file_ix_cur)
            time_ix.append(time_ix_cur)

        file_ix = np.concatenate(file_ix, axis=0)
        time_ix = np.concatenate(time_ix, axis=0)

        speaker = []
        for f in self.fileIDs:
            speaker.append(self.data[f].speaker)
        speaker = np.array(speaker)

        if forced_boundaries:
            gold_boundaries_tmp, _ = self.one_hot_boundaries(
                inner_segments=forced_boundaries,
                outer_segments=mask,
                padding=None
            )
            gold_boundaries = []
            for i, x in enumerate(gold_boundaries_tmp):
                gold_boundaries.append(x[0])
            gold_boundaries = pad_sequence(gold_boundaries, padding='post')
        else:
            gold_boundaries = None

        cache_dict = {
            'type': 'streaming',
            'n': n,
            'file_lengths': file_lengths,
            'feats_inputs': feats_inputs,
            'feats_targets': feats_targets,
            'boundaries': boundaries,
            'file_ix': file_ix,
            'time_ix': time_ix,
            'speaker': speaker,
            'forced_boundaries': gold_boundaries,
            'window_len_input': window_len_input,
            'window_len_bwd': window_len_bwd,
            'window_len_fwd': window_len_fwd,
            'target_bwd_resampling': target_bwd_resampling,
            'target_fwd_resampling': target_fwd_resampling,
            'predict_deltas': predict_deltas,
        }

        self.cache[name] = cache_dict

    def cache_files_data(
            self,
            name,
            mask,
            normalize_inputs=False,
            center_inputs=False
    ):
        feats, boundaries, _ = self.features(
            mask=mask,
            normalize=normalize_inputs,
            center=center_inputs
        )

        file_lengths = [len(x[0]) for x in feats]
        n = len(file_lengths)

        speaker = []
        for f in self.fileIDs:
            speaker.append(self.data[f].speaker)
        speaker = np.array(speaker)

        cache_dict = {
            'type': 'files',
            'n': n,
            'file_lengths': file_lengths,
            'feats': feats,
            'fixed_boundaries': boundaries,
            'speaker': speaker
        }

        self.cache[name] = cache_dict

    def get_streaming_indices(
            self,
            time_ix,
            window_len_input,
            window_len_bwd,
            window_len_fwd
    ):
        history_ix = time_ix[..., None] - window_len_input + np.arange(window_len_input)[None, ...] + 1
        bwd_context_ix = time_ix[..., None] - np.arange(window_len_bwd)[None, ...]
        fwd_context_ix = time_ix[..., None] + np.arange(1, window_len_fwd + 1)[None, ...]

        return history_ix, bwd_context_ix, fwd_context_ix

    def get_data_feed(
        self,
        name,
        minibatch_size=None,
        randomize=True
    ):
        if self.cache[name]['type'] == 'utterance':
            return self.get_utterance_data_feed(name, minibatch_size, randomize=randomize)
        elif self.cache[name]['type'] == 'streaming':
            return self.get_streaming_data_feed(name, minibatch_size, randomize=randomize)
        elif self.cache[name]['type'] == 'files':
            return self.get_files_data_feed(name, randomize=randomize)
        else:
            raise ValueError('Unrecognized data feed type: "%s".' % self.cache[name]['type'])

    def get_utterance_data_feed(
            self,
            name,
            minibatch_size,
            randomize=False
    ):
        n = self.cache[name]['n']
        i = 0
        if randomize:
            ix, ix_inv = get_random_permutation(n)
        else:
            ix = np.arange(n)

        X = self.cache[name]['X']
        X_mask = self.cache[name]['X_mask']
        y = self.cache[name]['y']
        y_mask = self.cache[name]['y_mask']
        speaker = self.cache[name]['speaker']
        gold_boundaries = self.cache[name]['gold_boundaries']

        while i < n:
            indices = ix[i:i + minibatch_size]

            out = {
                'X': X[indices],
                'X_mask': X_mask[indices],
                'y': y[indices],
                'y_mask': y_mask[indices],
                'speaker': speaker[indices],
                'gold_boundaries': None if gold_boundaries is None else gold_boundaries[indices],
                'indices': indices
            }

            i += minibatch_size

            yield out

    def get_streaming_data_feed(
            self,
            name,
            minibatch_size=1,
            randomize=True
    ):
        n = self.cache[name]['n']
        feats_inputs = self.cache[name]['feats_inputs']
        feats_targets = self.cache[name]['feats_targets']
        fixed_boundaries = self.cache[name]['boundaries']
        file_ix = self.cache[name]['file_ix']
        time_ix = self.cache[name]['time_ix']
        speaker = self.cache[name]['speaker']
        forced_boundaries = self.cache[name]['forced_boundaries']
        window_len_input = self.cache[name]['window_len_input']
        window_len_bwd = self.cache[name]['window_len_bwd']
        window_len_fwd = self.cache[name]['window_len_fwd']
        target_bwd_resampling = self.cache[name]['target_bwd_resampling']
        target_fwd_resampling = self.cache[name]['target_fwd_resampling']
        predict_deltas = self.cache[name]['predict_deltas']

        i = 0
        if randomize:
            ix, ix_inv = get_random_permutation(n)
        else:
            ix = np.arange(n)

        while i < n:
            indices = ix[i:i+minibatch_size]

            file_ix_cur = file_ix[indices]
            time_ix_cur = time_ix[indices]

            history_ix, bwd_context_ix, fwd_context_ix = self.get_streaming_indices(
                time_ix_cur,
                window_len_input,
                window_len_bwd,
                window_len_fwd
            )

            X_cur = feats_inputs[file_ix_cur[..., None], history_ix]

            if self.datatype == 'acoustic':
                if predict_deltas:
                    frame_slice = slice(None)
                else:
                    frame_slice = slice(0, self.n_coef)
            else:
                frame_slice = slice(0, len(self.ix2char))

            if forced_boundaries is not None:
                forced_boundaries_cur = forced_boundaries[file_ix_cur[..., None], history_ix]
            else:
                forced_boundaries_cur = None

            y_bwd_cur = feats_targets[file_ix_cur[..., None], bwd_context_ix, frame_slice]
            if target_bwd_resampling:
                y_bwd_cur = scipy.signal.resample(y_bwd_cur, target_bwd_resampling, axis=1)
                y_bwd_mask_cur = np.ones(y_bwd_cur.shape[:-1])
            else:
                y_bwd_mask_cur = np.any(y_bwd_cur, axis=-1)

            y_fwd_cur = feats_targets[file_ix_cur[..., None], fwd_context_ix, frame_slice]
            if target_fwd_resampling:
                y_fwd_cur = scipy.signal.resample(y_fwd_cur, target_fwd_resampling, axis=1)
                y_fwd_mask_cur = np.ones(y_fwd_cur.shape[:-1])
            else:
                y_fwd_mask_cur = np.any(y_fwd_cur, axis=-1)

            out = {
                'X': X_cur,
                'X_mask': np.any(X_cur, axis=-1),
                'forced_boundaries': forced_boundaries_cur,
                'fixed_boundaries': fixed_boundaries[file_ix_cur[..., None], history_ix],
                'y_bwd': y_bwd_cur,
                'y_bwd_mask': y_bwd_mask_cur,
                'y_fwd': y_fwd_cur,
                'y_fwd_mask': y_fwd_mask_cur,
                'speaker': speaker[file_ix_cur]
            }

            i += minibatch_size

            yield out

    def get_files_data_feed(
            self,
            name,
            randomize=True
    ):
        n = self.cache[name]['n']
        feats = self.cache[name]['feats']
        fixed_boundaries = self.cache[name]['fixed_boundaries']
        speaker = self.cache[name]['speaker']

        i = 0
        if randomize:
            ix, ix_inv = get_random_permutation(n)
        else:
            ix = np.arange(n)

        while i < n:
            index = ix[i]

            out = {
                'X': feats[index],
                'fixed_boundaries': fixed_boundaries[index],
                'speaker': speaker[index:index+1],
            }

            i += 1

            yield out

    def get_n(self, name):
        return self.cache[name]['n']

    def get_n_minibatch(self, name, minibatch_size):
        return math.ceil(self.get_n(name) / minibatch_size)

    def cached(self, name):
        return name in self.cache

    def as_text(self):
        if self.datatype.lower() == 'text':
            out = ''
            for f in self.fileIDs:
                out += self.data[f].as_text()
        else:
            sys.stderr.write('Converting to text not supported for acoustic-type data. Skipping...\n')
            out = None

        return out

    def inputs(
            self,
            segments='vad',
            max_len=None,
            padding='pre',
            reverse=False,
            normalize=False,
            center=False,
            resample=None
    ):
        return self.segment_and_stack(
            segments=segments,
            max_len=max_len,
            padding=padding,
            reverse=reverse,
            normalize=normalize,
            center=center,
            resample = resample
        )

    def targets(
            self,
            segments='vad',
            padding='post',
            max_len=None,
            reverse=True,
            normalize=False,
            center=False,
            with_deltas=False,
            resample=None
    ):
        return self.segment_and_stack(
            segments=segments,
            max_len=max_len,
            padding=padding,
            reverse=reverse,
            normalize=normalize,
            center=center,
            with_deltas=with_deltas,
            resample=resample
        )

    def one_hot_boundaries(
            self,
            inner_segments='wrd',
            outer_segments='vad',
            padding='post',
            max_len=None,
            reverse=False
    ):
        features = {}
        for f in self.fileIDs:
            features[f] = self.data[f].one_hot_boundaries(segments=inner_segments)[..., None]

        return self.segment_and_stack(
            features=features,
            boundaries_as_features=True,
            segments=outer_segments,
            max_len=max_len,
            padding=padding,
            reverse=reverse
        )

    def segments(self, segment_type='vad'):
        return pd.concat([self.data[f].segments(segment_type=segment_type) for f in self.fileIDs], axis=0)

    def align_gold_labels_to_segments(self, true, pred):
        out = []

        for f in self.fileIDs:
            if isinstance(true, str):
                true_cur = true
            else:
                # ``true`` is a dictionary
                true_cur = true[f]

            if isinstance(pred, str):
                pred_cur = pred
            else:
                pred_cur = pred[f]

            out.append(self.data[f].align_gold_labels_to_segments(true_cur, pred_cur))

        return pd.concat(out, axis=0)

    def ix2label(self, segment_type='wrd'):
        labels = self.segments(segment_type=segment_type).label
        ix2label = sorted(list(labels.unique()))

        return ix2label

    def label2ix(self, segment_type='wrd'):
        ix2label = self.ix2label(segment_type=segment_type)

        label2ix = {}
        for ix in range(len(ix2label)):
            label2ix[ix2label[ix]] = ix

    def n_classes(self, segment_type='wrd'):
        return len(self.ix2label(segment_type=segment_type))

    def labels(self, one_hot=True, segment_type='wrd', segments=None):
        if segments is None:
            segments = self.segments(segment_type=segment_type)

        labels = segments.label

        ix2label = self.ix2label(segment_type=segment_type)

        label2ix = {}
        for ix in range(len(ix2label)):
            label2ix[ix2label[ix]] = ix

        if one_hot:
            out = np.zeros((len(labels), len(ix2label)))
            out[np.arange(len(labels)), labels.apply(lambda x: label2ix[x])] = 1
        else:
            out = np.array(labels.apply(lambda x: label2ix[x]), dtype=np.float32)

        return out

    def file_indices(self, segment_type='vad'):
        assert segment_type in ['vad', 'phn', 'wrd'], 'Only segment types "vad", "phn", and "wrd" are supported for file index extraction'

        n = 0
        file_indices = {}
        for f in self.fileIDs:
            if segment_type == 'vad':
                file_len = len(self.data[f].vad_segments)
            elif segment_type == 'wrd':
                file_len = len(self.data[f].wrd_segments)
            else:
                file_len = len(self.data[f].phn_segments)
            file_indices[f] = (n, n + file_len)
            n += file_len

        return file_indices

    def shuffle(self, inputs, segment_type='vad', reshuffle=False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out = []

        for d in inputs:
            if segment_type == 'vad':
                if self.vad_perm is None or reshuffle:
                    perm = np.random.permutation(np.arange(len(d)))
                    perm_inv = np.zeros_like(perm)
                    perm_inv[perm] = np.arange(len(d))

                    self.vad_perm = perm
                    self.vad_perm_inv = perm_inv

                out.append(d[self.vad_perm])

            if segment_type == 'wrd':
                if self.wrd_perm is None or reshuffle:
                    perm = np.random.permutation(np.arange(len(d)))
                    perm_inv = np.zeros_like(perm)
                    perm_inv[perm] = np.arange(len(d))

                    self.wrd_perm = perm
                    self.wrd_perm_inv = perm_inv

                out.append(d[self.wrd_perm])

            if segment_type == 'phn':
                if self.phn_perm is None or reshuffle:
                    perm = np.random.permutation(np.arange(len(d)))
                    perm_inv = np.zeros_like(perm)
                    perm_inv[perm] = np.arange(len(d))

                    self.phn_perm = perm
                    self.phn_perm_inv = perm_inv

                out.append(d[self.phn_perm])

        if len(out) == 1:
            out = out[0]

        return out

    def deshuffle(self, inputs, segment_type='vad'):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out = []

        for d in inputs:
            if segment_type == 'vad':
                assert self.vad_perm is not None, 'Attempting to deshuffle before shuffling has been initiated'
                out.append(d[self.vad_perm_inv])

            if segment_type == 'wrd':
                assert self.wrd_perm is not None, 'Attempting to deshuffle before shuffling has been initiated'
                out.append(d[self.wrd_perm_inv])

            if segment_type == 'phn':
                assert self.phn_perm is not None, 'Attempting to deshuffle before shuffling has been initiated'
                out.append(d[self.phn_perm_inv])

        if len(out) == 1:
            out = out[0]

        return out

    def initialize_random_segmentation(self, rate):
        for f in self.data:
            self.data[f].initialize_random_segmentation(rate)

    def get_segment_tables_from_segmenter_states(
            self,
            segmentations,
            parent_segment_type='vad',
            states=None,
            discretize=True,
            state_activation='tanh',
            smoothing_algorithm=None,
            smoothing_algorithm_params=None,
            seconds_per_step=None,
            n_points=None,
            mask=None,
            padding=None
    ):
        if seconds_per_step is None:
            seconds_per_step = self.seconds_per_step

        n_levels = len(segmentations)
        out = []

        for i in range(n_levels):
            out.append({})

        i = 0

        for f in self.fileIDs:
            F = self.data[f]
            if parent_segment_type == 'vad':
                n_utt = len(F.vad_segments)
            elif parent_segment_type == 'wrd':
                n_utt = len(F.wrd_segments)
            elif parent_segment_type == 'phn':
                n_utt = len(F.phn_segments)
            elif parent_segment_type == 'rnd':
                n_utt = len(F.rnd_segments)

            dfs = F.get_segment_tables_from_segmenter_states(
                [s[i:i+n_utt] for s in segmentations],
                parent_segment_type=parent_segment_type,
                states=[s[i:i+n_utt] for s in states],
                discretize=discretize,
                mask=None if mask is None else mask[i:i + n_utt],
                state_activation=state_activation,
                smoothing_algorithm=smoothing_algorithm,
                smoothing_algorithm_params=smoothing_algorithm_params,
                seconds_per_step=seconds_per_step,
                n_points=n_points,
                padding=padding
            )

            for j in range(n_levels):
                if len(dfs) > 0:
                    out[j][f] = dfs[j]

            i += n_utt

        return out

    def score_segmentation(self, true, pred, tol=0.02):
        score_dict = {}
        for f in self.fileIDs:
            if isinstance(true, str):
                true_cur = self.data[f].segments(true)
            else:
                # ``true`` is a dictionary
                true_cur = true[f]

            if isinstance(pred, str):
                pred_cur = self.data[f].segments(pred)
            else:
                # ``true`` is a dictionary
                if f in pred:
                    pred_cur = pred[f]
                else:
                    pred_cur = None

            if pred_cur is None:
                fn = len(true_cur)
                score_dict[f] = {
                    'b_tp': 0.,
                    'b_fp': 0.,
                    'b_fn': fn,
                    'w_tp': 0.,
                    'w_fp': 0.,
                    'w_fn': fn,
                }
            else:
                score_dict[f] = self.data[f].score_segmentation(true_cur, pred_cur, tol=tol)

        global_score_dict = {
            'b_tp': sum([score_dict[f]['b_tp'] for f in self.fileIDs]),
            'b_fp': sum([score_dict[f]['b_fp'] for f in self.fileIDs]),
            'b_fn': sum([score_dict[f]['b_fn'] for f in self.fileIDs]),
            'w_tp': sum([score_dict[f]['w_tp'] for f in self.fileIDs]),
            'w_fp': sum([score_dict[f]['w_fp'] for f in self.fileIDs]),
            'w_fn': sum([score_dict[f]['w_fn'] for f in self.fileIDs]),
        }

        return global_score_dict, score_dict

    def score_text_segmentation(self, true, pred):
        score_dict = {}
        for f in self.fileIDs:
            if isinstance(true, str):
                true_cur = self.data[f].segments(true)
            else:
                # ``true`` is a dictionary
                true_cur = true[f]

            if isinstance(pred, str):
                pred_cur = self.data[f].segments(pred)
            else:
                # ``true`` is a dictionary
                if f in pred:
                    pred_cur = pred[f]
                else:
                    pred_cur = None

            if pred_cur is None:
                fn = len(true_cur)
                score_dict[f] = {
                    'b_tp': 0.,
                    'b_fp': 0.,
                    'b_fn': fn,
                    'w_tp': 0.,
                    'w_fp': 0.,
                    'w_fn': fn,
                    'l_tp': 0.,
                    'l_fp': 0.,
                    'l_fn': fn,
                }
            else:
                score_dict[f] = self.data[f].score_text_segmentation(true_cur, pred_cur)

        global_score_dict = {
            'b_tp': sum([score_dict[f]['b_tp'] for f in self.fileIDs]),
            'b_fp': sum([score_dict[f]['b_fp'] for f in self.fileIDs]),
            'b_fn': sum([score_dict[f]['b_fn'] for f in self.fileIDs]),
            'w_tp': sum([score_dict[f]['w_tp'] for f in self.fileIDs]),
            'w_fp': sum([score_dict[f]['w_fp'] for f in self.fileIDs]),
            'w_fn': sum([score_dict[f]['w_fn'] for f in self.fileIDs]),
            'l_tp': sum([score_dict[f]['l_tp'] for f in self.fileIDs]),
            'l_fp': sum([score_dict[f]['l_fp'] for f in self.fileIDs]),
            'l_fn': sum([score_dict[f]['l_fn'] for f in self.fileIDs]),
        }

        return global_score_dict, score_dict


    def segmentations_to_string(self, segments, parent_segments=None):
        if self.datatype == 'text':
            out = ''
            for f in self.fileIDs:
                out += self.data[f].segmentations_to_string(segments, parent_segments=parent_segments)

        else:
            sys.stderr.write('Converting data to string not supported for acoustic-type datasets. Skipping...\n')
            out = None

        return out

    def dump_segmentations_to_textgrid(self, outdir=None, suffix='', segments=None):
        if self.datatype == 'acoustic':
            for f in self.fileIDs:
                segments_cur = []
                for seg in segments:
                    if isinstance(seg, str):
                        segments_cur.append(seg)
                    else:
                        segments_cur.append(seg[f])

                self.data[f].dump_segmentations_to_textgrid(
                    outdir=outdir,
                    suffix=suffix,
                    segments=segments_cur
                )
        else:
            sys.stderr.write('Dumping to textgrid not supported for text-type datasets. Skipping...\n')

    def dump_segmentations_to_textfile(self, outdir=None, suffix='', segments=None, parent_segments=None):
        if self.datatype == 'text':
            for f in self.fileIDs:
                segments_cur = []
                for seg in segments:
                    if isinstance(seg, str):
                        segments_cur.append(seg)
                    else:
                        segments_cur.append(seg[f])
                if isinstance(parent_segments, str):
                    parent_segments_cur = parent_segments
                else:
                    parent_segments_cur = parent_segments[f]

                self.data[f].dump_segmentations_to_textfile(
                    outdir=outdir,
                    suffix=suffix,
                    segments=segments_cur,
                    parent_segments=parent_segments_cur
                )
        else:
            sys.stderr.write('Dumping to text file not supported for acoustic-type datasets. Skipping...\n')

    def summary(self, indent=0, summarize_components=False):
        out = ' ' * indent + 'DATASET SUMMARY:\n\n'
        out += ' ' * indent + 'Data type: %s\n' % self.datatype
        out += ' ' * (indent + 2) + 'Source directory: %s\n' %self.dir_path
        length = sum([len(self.data[f]) for f in self.data])

        if self.datatype == 'acoustic':
            duration = sum([self.data[f].duration for f in self.data])
            out += ' ' * (indent + 2) + 'Total duration (seconds): %.4f\n' %duration
            out += ' ' * (indent + 2) + 'Total length (frames): %.4f\n' %length
            out += ' ' * (indent + 2) + 'Sampling rate: %s\n' %self.sr
            out += ' ' * (indent + 2) + 'Frame length: %sms\n' %self.window_len
            out += ' ' * (indent + 2) + 'Step size: %sms\n' %self.offset
            out += ' ' * (indent + 2) + 'Number of cepstral coefficients: %s\n' %self.n_coef
            out += ' ' * (indent + 2) + 'Number of derivatives: %s\n' %self.order
        else:
            out += ' ' * (indent + 2) + 'Total length (characters): %.4f\n' %length

        if self.clip_timesteps:
            out += ' ' * (indent + 2) + 'Frame clipping limit: %s\n' %self.clip_timesteps

        out += '\n'
        vad_segments = self.segments('vad')
        out += ' ' * (indent + 2) + 'Number of VAD segments: %s\n' % len(vad_segments)
        out += segment_length_summary(vad_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        out += '\n'
        phn_segments = self.segments('phn')
        out += ' ' * (indent + 2) + 'Number of phone segments: %s\n' % len(phn_segments)
        out += ' ' * (indent + 2) + 'Number of phone types: %s\n' % len(phn_segments.label.unique())
        out += segment_length_summary(phn_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        out += '\n'
        wrd_segments = self.segments('wrd')
        out += ' ' * (indent + 2) + 'Number of word segments: %s\n' % len(wrd_segments)
        out += ' ' * (indent + 2) + 'Number of word types: %s\n' % len(wrd_segments.label.unique())
        out += segment_length_summary(wrd_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        if summarize_components:
            out += '\n' + ' ' * (indent + 2) + 'COMPONENT DATA FILE SUMMARIES:\n\n'
            for f in self.data:
                out += self.data[f].summary(indent=indent+4, report_metadata=False)
                out += '\n'

        return out


class Datafile(object):
    def __new__(cls, *args, **kwargs):
        if cls is Datafile:
            raise TypeError("Datafile is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(
            self,
            path,
            clip_timesteps=None
    ):
        self.clip_timesteps = clip_timesteps

        self.ID = os.path.basename(path)[:-4]
        self.dir = os.path.dirname(path)
        self.text_path = path

        self.vad_segments = None
        self.phn_segments = None
        self.wrd_segments = None

        self.data_src = None

    def __len__(self):
        return self.len

    def data(self):
        raise NotImplementedError

    def segments(self, segment_type='vad'):
        if isinstance(segment_type, str):
            if segment_type == 'vad':
                return self.vad_segments
            if segment_type == 'wrd':
                return self.wrd_segments
            if segment_type == 'phn':
                return self.phn_segments
            if segment_type == 'rnd':
                return self.rnd_segments
            raise ValueError('Unrecognized segment type name "%s".' % segment_type)
        return segment_type

    def align_gold_labels_to_segments(self, true, pred):
        if isinstance(true, str):
            true = self.segments(segment_type=true)
        if isinstance(pred, str):
            pred = self.segments(sgement_type=pred)

        aligned = compute_unsupervised_classification_targets(true, pred)

        return aligned

    def segments_to_one_hot(self, segments):
        one_hot = np.zeros(len(self))
        ix = np.array(segments.end - 1e-8) # epsilon guarantees no exact integer values of ix, which can introduce a fencepost problem
        ix *= self.steps_per_second
        ix = ix.astype('int')
        one_hot[ix] = 1

        return one_hot

    def one_hot_boundaries(self, segments='vad'):
        segments = self.segments(segments)
        one_hot = self.segments_to_one_hot(segments)

        return one_hot

    def segment_and_stack(
            self,
            features=None,
            boundaries_as_features=False,
            segments='vad',
            max_len=None,
            padding='pre',
            pad_final_one=False,
            reverse=False,
            normalize=False,
            center=False,
            with_deltas=True,
            resample=None
    ):
        if features is None:
            feats = self.data()

            if normalize:
                maximum = feats.max()
                minimum = feats.min()
                diff = maximum - minimum
                feats = (feats - minimum) / diff
            if center:
                feats = feats - feats.mean()
        else:
            feats = features

        if isinstance(segments, str):
            if segments == 'vad':
                segments_arr = self.vad_segments
            elif segments == 'wrd':
                segments_arr = self.wrd_segments
            elif segments == 'phn':
                segments_arr = self.phn_segments
            elif segments == 'rnd':
                segments_arr = self.rnd_segments
            else:
                raise ValueError('Segment type "%s" not recognized.' %segments)
        else:
            segments_arr = segments

        pad_seqs = padding not in ['None', None]

        bounds_left = np.floor(segments_arr.start * self.steps_per_second)
        bounds_right = np.ceil(segments_arr.end * self.steps_per_second)
        bounds = np.stack([bounds_left, bounds_right], axis=1).astype(np.int32)

        feats_split = []
        mask_split = []
        for i in range(len(bounds)):
            s = bounds[i, 0]
            e = bounds[i, 1]
            if max_len is not None:
                e = min(e, s + max_len)
            new_feats = feats[s:e]

            if boundaries_as_features:
                # Returning 1-hot segments, not acoustics, insert final boundary
                new_feats[-1] = 1

            length = e - s
            if resample and new_feats.shape[0] > 0:
                length = resample
                new_feats = scipy.signal.resample(new_feats, resample, axis=0)

            if not with_deltas:
                new_feats = new_feats[:,:self.n_coef]

            if not pad_seqs:
                new_feats = new_feats[None, ...]
                mask = np.ones((1, length))
            else:
                mask = np.ones(length)

            feats_split.append(new_feats)
            mask_split.append(mask)

        if pad_seqs:
            feats = pad_sequence(feats_split, padding=padding, reverse=reverse)
            mask = pad_sequence(mask_split, padding=padding, reverse=reverse)
        else:
            feats = feats_split
            mask = mask_split

        return feats, mask

    def generate_random_segmentation(self, mean_frames_per_segment, parent_segment_type='vad'):
        if parent_segment_type == 'vad':
            parent_segments = self.vad_segments
        elif parent_segment_type == 'wrd':
            parent_segments = self.wrd_segments
        elif parent_segment_type == 'phn':
            parent_segments = self.phn_segments
        elif parent_segment_type == 'rnd':
            parent_segments = self.rnd_segments

        out_start = []
        out_end = []
        for x in parent_segments[['start', 'end']].as_matrix():
            s_p, e_p = x
            s = s_p
            e = s
            while e < e_p:
                e = min(e + np.random.uniform(0., mean_frames_per_segment * self.seconds_per_step * 2), e_p)
                out_start.append(s)
                out_end.append(e)
                s = e

        out = {
            'start': out_start,
            'end': out_end,
            'label': 0
        }

        out = pd.DataFrame(out)

        return out

    def initialize_random_segmentation(self, rate):
        self.rnd_segments = self.generate_random_segmentation(rate)

    def get_segment_tables_from_segmenter_states(
            self,
            segmentations,
            parent_segment_type='vad',
            states=None,
            discretize=True,
            state_activation='tanh',
            smoothing_algorithm=None,
            smoothing_algorithm_params=None,
            seconds_per_step=None,
            n_points=None,
            mask=None,
            padding=None,
            snap_ends=True
    ):
        if seconds_per_step is None:
            seconds_per_step = self.seconds_per_step

        if parent_segment_type == 'vad':
            parent_segs = self.vad_segments
        elif parent_segment_type == 'wrd':
            parent_segs = self.wrd_segments
        elif parent_segment_type == 'phn':
            parent_segs = self.phn_segments
        elif parent_segment_type == 'rnd':
            parent_segs = self.rnd_segments

        out = []

        parent_starts = parent_segs.start.as_matrix()
        parent_ends = parent_segs.end.as_matrix()

        n_layers = len(segmentations)

        if len(segmentations[0]) > 0:
            for i in range(n_layers):
                starts = []
                ends = []
                labels = []

                timestamps = extract_segment_timestamps_batch(
                    segmentations[i],
                    algorithm=smoothing_algorithm,
                    algorithm_params=smoothing_algorithm_params,
                    seconds_per_step=seconds_per_step,
                    n_points=n_points,
                    mask=mask,
                    padding=padding
                )

                if states is not None:
                    states_extracted = extract_states_at_timestamps_batch(
                        timestamps,
                        states[i],
                        steps_per_second=seconds_per_step,
                        activation=state_activation,
                        discretize=discretize,
                        as_categories=False,
                        as_onehot=True,
                        mask=mask,
                        padding=padding
                    )

                for j, s in enumerate(timestamps):
                    s = np.concatenate([[0.], s], axis=0)
                    segs = s + parent_starts[j]
                    starts_cur = segs[:-1]
                    ends_cur = segs[1:]
                    if snap_ends:
                        ends_cur[-1] = parent_ends[j]
                    starts.append(starts_cur)
                    ends.append(ends_cur)
                    if states is not None:
                        labels.append(states_extracted[j])

                starts = np.concatenate(starts, axis=0)
                ends = np.concatenate(ends, axis=0)

                if states is not None:
                    labels = np.concatenate(labels, axis=0)
                else:
                    labels = 0

                df = {
                    'start': starts,
                    'end': ends
                }

                if discretize:
                    df['label'] = labels
                else:
                    df['label'] = 0
                    for j in range(labels.shape[1]):
                        df['d%s' %j] = labels[:,j]

                df = pd.DataFrame(df)

                out.append(df)
        else:
            df = pd.DataFrame(columns=['start', 'end', 'label'])

        return out

    def indicator2segs(self, indicator, mask=None, mask_starts=None, mask_ends=None, labels=None, location='start'):
        start = np.zeros_like(indicator)
        end = np.zeros_like(indicator)
        if location == 'start':
            start += indicator
            end[:-1] += indicator[1:]

        else:
            start[1:] += indicator[:-1]
            end += indicator

        start[0] = 1
        if mask is not None:
            start = (start + mask_starts) * mask
        start = np.where((np.arange(len(start))) * start)[0] * self.seconds_per_step

        end[-1] = 1
        if mask is not None:
            end = (end + mask_ends) * mask
        end = (np.where((np.arange(len(end))) * end)[0] + 1) * self.seconds_per_step

        out = {
            'start': start,
            'end': end,
            'speaker': 0,
            'index': 0
        }

        if labels is None:
            out['label'] = 0
        else:
            out['label'] = labels

        out = pd.DataFrame(out)

        return out

    def segs2indicator(self, segs, location='start'):
        out = np.zeros(len(self))
        if location == 'start':
            ix = np.floor(segs.start * self.steps_per_second).astype('int')
        else:
            ix = np.floor(segs.end * self.steps_per_second).astype('int') - 1

        out[ix] = 1.

        return out

    def segs2mask(self, segs, max_len=None):
        out = np.zeros(len(self))

        bounds_left = np.floor(segs.start * self.steps_per_second)
        bounds_right = np.ceil(segs.end * self.steps_per_second)
        bounds = np.stack([bounds_left, bounds_right], axis=1).astype(np.int32)

        for i in range(len(bounds)):
            s = bounds[i, 0]
            e = bounds[i, 1]
            if max_len is not None:
                e = min(e, s + max_len)
            out[s:e] = 1

        return out

    def score_segmentation(self, true, pred, tol=0.02):
        if isinstance(true, str):
            if true == 'vad':
                true = self.vad_segments
            elif true == 'wrd':
                true = self.wrd_segments
            elif true == 'phn':
                true = self.phn_segments
            elif true == 'rnd':
                true = self.rnd_segments

        if isinstance(pred, str):
            if pred == 'vad':
                pred = self.vad_segments
            elif pred == 'wrd':
                pred = self.wrd_segments
            elif pred == 'phn':
                pred = self.phn_segments
            elif pred == 'rnd':
                pred = self.rnd_segments

        score_dict = score_segmentation(true, pred, tol=tol)

        return score_dict


class AcousticDatafile(Datafile):
    def __init__(
            self,
            path,
            sr = 16000,
            offset = 10,
            window_len = 25,
            n_coef = 13,
            order = 2,
            clip_timesteps=None,
    ):
        super(AcousticDatafile, self).__init__(path, clip_timesteps=clip_timesteps)

        assert path.lower().endswith('.wav'), 'Input file "%s" was not .wav.' %path
        assert sr % 1000 == 0, 'Must use a sampling rate that is a multiple of 1000'

        self.sr = sr
        self.offset = offset
        self.window_len = window_len
        self.n_coef = n_coef
        self.order = order
        self.speaker = None

        data_src, duration = wav_to_mfcc(
            path,
            sr=sr,
            offset=offset,
            window_len=window_len,
            n_coef=n_coef,
            order=order
        )
        data_src = np.transpose(data_src, [1, 0])

        if self.clip_timesteps is not None:
            data_src = data_src[:clip_timesteps, :]

        self.data_src = data_src
        self.shape = self.data_src.shape

        self.len = data_src.shape[0]
        self.duration = duration

        vad_path = self.dir + '/' + self.ID + '.vad'
        if os.path.exists(vad_path):
            self.vad_path = vad_path
            self.vad_segments = pd.read_csv(self.vad_path, sep=' ')
            self.vad_segments.speaker = self.vad_segments.speaker.astype('str')
            self.vad_segments.sort_values('start', inplace=True)
            self.vad_segments['label'] = 0
            if self.speaker is None and 'speaker' in self.vad_segments.columns:
                self.speaker = self.vad_segments.speaker.iloc[0]
        else:
            self.vad_path = None
            self.vad_segments = pd.DataFrame(
                {
                    'start': [0.],
                    'end': [float(self.len)],
                    'label': '0',
                    'speaker': '0',
                    'index': 0
                }
            )
        self.vad_segments.fileID = self.ID

        phn_path = self.dir + '/' + self.ID + '.phn'
        if os.path.exists(phn_path):
            self.phn_path = phn_path
            self.phn_segments = pd.read_csv(self.phn_path, sep=' ')
            self.phn_segments.speaker = self.phn_segments.speaker.astype('str')
            self.phn_segments.sort_values('start', inplace=True)
            if self.speaker is None and 'speaker' in self.phn_segments.columns:
                self.speaker = self.phn_segments.speaker.iloc[0]
        else:
            self.phn_path = None
            if self.vad_path is None:
                self.phn_segments = pd.DataFrame(
                    {
                        'start': [0.],
                        'end': [float(self.len)],
                        'label': '0',
                        'speaker': '0',
                        'index': 0
                    }
                )
            else:
                self.phn_segments = self.vad_segments
        self.phn_segments.fileID = self.ID

        wrd_path = self.dir + '/' + self.ID + '.wrd'
        if os.path.exists(wrd_path):
            self.wrd_path = wrd_path
            self.wrd_segments = pd.read_csv(self.wrd_path, sep=' ')
            self.wrd_segments.speaker = self.wrd_segments.speaker.astype('str')
            self.wrd_segments.sort_values('start', inplace=True)
            if self.speaker is None and 'speaker' in self.wrd_segments.columns:
                self.speaker = self.wrd_segments.speaker.iloc[0]
        else:
            self.wrd_path = None
            if self.vad_path is None:
                self.wrd_segments = pd.DataFrame(
                    {
                        'start': [0.],
                        'end': [float(self.len)],
                        'label': 0,
                        'speaker': 0,
                        'index': 0
                    }
                )
            else:
                self.wrd_segments = self.vad_segments
        self.wrd_segments.fileID = self.ID

        self.rnd_segments = None

        self.steps_per_second = 1000. / self.offset
        self.seconds_per_step = float(self.offset) / 1000.

    def data(self):
        return self.data_src

    def dump_segmentations_to_textgrid(self, outdir=None, suffix='', segments=None):
        if outdir is None:
            outdir = self.dir

        if not isinstance(segments, list):
            segments = [segments]

        path = outdir + '/' + self.ID + '_segmentations' + suffix + '.TextGrid'
        with open(path, 'w') as f:
            for i, seg in enumerate(segments):
                if isinstance(seg, str):
                    segment_type = seg
                    seg = self.segments(segment_type)
                else:
                    segment_type = 'pred'

                if i == 0:
                    f.write('File type = "ooTextFile"\n')
                    f.write('Object class = "TextGrid"\n')
                    f.write('\n')
                    f.write('xmin = %.3f\n' % seg.start.iloc[0])
                    f.write('xmax = %.3f\n' % seg.end.iloc[-1])
                    f.write('tiers? <exists>\n')
                    f.write('size = %d\n' %len(segments))
                    f.write('item []:\n')

                f.write('    item [%d]:\n' %i)
                f.write('        class = "IntervalTier"\n')
                f.write('        class = "segmentations_%s"\n' %segment_type)
                f.write('        xmin = %.3f\n' % seg.start.iloc[0])
                f.write('        xmax = %.3f\n' % seg.end.iloc[-1])
                f.write('        intervals: size = %d\n' % len(seg))

                row_str = '        intervals [%d]:\n' + \
                    '            xmin = %.3f\n' + \
                    '            xmax = %.3f\n' + \
                    '            text = "%s"\n\n'

                if 'label' in seg.columns:
                    for j, r in seg[['start', 'end', 'label']].iterrows():
                        f.write(row_str % (j, r.start, r.end, r.label))
                else:
                    for j, r in seg[['start', 'end']].iterrows():
                        f.write(row_str % (j, r.start, r.end, ''))

    def summary(self, indent=0, report_metadata=True):
        out = ' ' * indent + 'DATA FILE SUMMARY: %s\n\n' %self.ID
        out += ' ' * (indent + 2) + 'Source location: %s\n' %self.path
        out += ' ' * (indent + 2) + 'Duration (seconds): %.4f\n' %(self.duration)
        out += ' ' * (indent + 2) + 'Length (frames): %.4f\n' %len(self)

        if report_metadata:
            out += ' ' * (indent + 2) + 'Sampling rate: %s\n' %self.sr
            out += ' ' * (indent + 2) + 'Frame length: %sms\n' %self.window_len
            out += ' ' * (indent + 2) + 'Step size: %sms\n' %self.offset
            out += ' ' * (indent + 2) + 'Number of cepstral coefficients: %s\n' %self.n_coef
            out += ' ' * (indent + 2) + 'Number of derivatives: %s\n' %self.order

            if self.clip_timesteps:
                out += ' ' * (indent + 2) + 'Frame clipping limit: %s\n' %self.clip_timesteps

        if self.vad_path:
            out += '\n' + ' ' * (indent + 2) + 'VAD segmentation file location: %s\n' %self.vad_path
        else:
            out += '\n' + ' ' * (indent + 2) + 'No VAD segmentation file provided.\n'
        out += ' ' * (indent + 2) + 'Number of VAD segments: %s\n' % len(self.vad_segments)
        out += segment_length_summary(self.vad_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        if self.phn_path:
            out += '\n' + ' ' * (indent + 2) + 'Phone segmentation file location: %s\n' %self.phn_path
        else:
            out += '\n' + ' ' * (indent + 2) + 'No phone segmentation file provided.\n'
        out += ' ' * (indent + 2) + 'Number of phone segments: %s\n' % len(self.phn_segments)
        out += ' ' * (indent + 2) + 'Number of phone types: %s\n' % len(self.phn_segments.label.unique())
        out += segment_length_summary(self.phn_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        if self.wrd_path:
            out += '\n' + ' ' * (indent + 2) + 'Word segmentation file location: %s\n' %self.wrd_path
        else:
            out += '\n' + ' ' * (indent + 2) + 'No word segmentation file provided.\n'
        out += ' ' * (indent + 2) + 'Number of word segments: %s\n' % len(self.wrd_segments)
        out += ' ' * (indent + 2) + 'Number of word types: %s\n' % len(self.wrd_segments.label.unique())
        out += segment_length_summary(self.wrd_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        return out


class TextDatafile(Datafile):
    def __init__(
            self,
            path,
            lower=False,
            clip_timesteps=None,
            speaker=None,
    ):
        super(TextDatafile, self).__init__(path, clip_timesteps=clip_timesteps)

        if speaker is None:
            self.speaker = '0'
        else:
            self.speaker = speaker

        self.lower = lower

        data_src = []
        words = []
        word_start = []
        word_end = []
        utt_start = []
        utt_end = []
        with open(path, 'r') as f:
            for l in f.readlines():
                if l.strip():
                    line = l.strip()
                    if self.lower:
                        line = line.lower()
                    new_words = line.split()
                    utt_start.append(float(len(data_src)))
                    utt_end_new = float(len(data_src))
                    for j, w in enumerate(new_words):
                        words.append(w)
                        word_start.append(float(len(data_src)))
                        word_end.append(float(len(data_src)) + len(w))
                        utt_end_new += float(len(w))
                        data_src += w
                    utt_end.append(utt_end_new)

        self.data_src = data_src

        self.vad_segments = pd.DataFrame(
            {
                'start': utt_start,
                'end': utt_end,
                'label': range(len(utt_start)),
                'speaker': self.speaker,
                'index': range(len(utt_start))
            }
        )
        self.vad_segments['fileID'] = self.ID

        self.wrd_segments = pd.DataFrame(
            {
                'start': word_start,
                'end': word_end,
                'label': words,
                'speaker': self.speaker,
                'index': range(len(word_start))
            }
        )
        self.wrd_segments['fileID'] = self.ID

        self.phn_segments = pd.DataFrame(
            {
                'start': np.array(list(range(len(data_src))), dtype='float'),
                'end': np.array(list(range(1,len(data_src)+1)), dtype='float'),
                'label': data_src,
                'speaker': speaker if speaker else '0',
                'index': np.array(range(len(data_src)))
            }
        )
        self.phn_segments['fileID'] = self.ID

        self.ix2char = sorted(list(set(data_src)))
        self.len = len(data_src)

        self.offset = 1.
        self.seconds_per_step = 1.
        self.steps_per_second = 1.

    def data(self):
        n_char = len(self.ix2char)
        indices = []
        for c in self.data_src:
            indices.append(self.char2ix[c])
        data = np.zeros((len(self), n_char))
        data[np.arange(len(self)), indices] = 1

        return data

    def update_charset(self, ix2char, char2ix=None):
        assert isinstance(ix2char, list), 'New charset c must be of type ``list``.'
        self.ix2char = ix2char
        if char2ix is None:
            self.char2ix = {}
            for i, x in enumerate(self.ix2char):
                self.char2ix[i] = x
        else:
            self.char2ix = char2ix

    def decode_utterance(self, one_hot, boundaries=None):
        out = ''
        for i, x in enumerate(np.argmax(one_hot, axis=-1)):
            c = self.ix2char[x]
            if c != ' ':
                out += c
            if boundaries is not None:
                if boundaries[i] == 1:
                    out += ' '

        return out

    def segmentations_to_string(self, segments, parent_segments=None):
        segment_ends = self.one_hot_boundaries(segments=segments)
        if parent_segments is not None:
            utts, _ = self.segment_and_stack(
                segments=parent_segments,
                padding=None,
            )
            bounds, _ = self.segment_and_stack(
                features=segment_ends,
                boundaries_as_features=True,
                segments=parent_segments,
                padding=None,
            )
        else:
            utts = self.data()

        out = ''
        for i, u in enumerate(utts):
            out += self.decode_utterance(u[0], bounds[i][0]) + '\n'

        return out

    def as_text(self):
        out = self.segmentations_to_string('wrd', parent_segments='vad')

        return out

    def score_text_segmentation(self, true, pred):
        if isinstance(true, str):
            if true == 'vad':
                true = self.vad_segments
            elif true == 'wrd':
                true = self.wrd_segments
            elif true == 'phn':
                true = self.phn_segments
            elif true == 'rnd':
                true = self.rnd_segments

        true = self.segmentations_to_string(true, parent_segments='vad')

        if isinstance(pred, str):
            if pred == 'vad':
                pred = self.vad_segments
            elif pred == 'wrd':
                pred = self.wrd_segments
            elif pred == 'phn':
                pred = self.phn_segments
            elif pred == 'rnd':
                pred = self.rnd_segments

        pred = self.segmentations_to_string(pred, parent_segments='vad')

        score_dict = score_text_segmentation(true, pred)

        return score_dict

    def dump_segmentations_to_textfile(self, segments, outdir=None, suffix='', parent_segments=None):
        if outdir is None:
            outdir = self.dir

        for i, seg in enumerate(segments):
            if isinstance(seg, str):
                segment_type = seg
                seg = self.segments(seg)
            else:
                segment_type = 'pred'

            out = self.segmentations_to_string(seg, parent_segments=parent_segments)

            path = outdir + '/' + self.ID + '_segmentations_' + segment_type + '_' + suffix + '.txt'

            with open(path, 'w') as f:
                f.write(out)

    def summary(self, indent=0, report_metadata=True):
        out = ' ' * indent + 'DATA FILE SUMMARY: %s\n\n' % self.ID
        out += ' ' * (indent + 2) + 'Source location: %s\n' % self.path
        out += ' ' * (indent + 2) + 'Length (characters): %.4f\n' % len(self)

        if report_metadata:
            if self.clip_timesteps:
                out += ' ' * (indent + 2) + 'Frame clipping limit: %s\n' % self.clip_timesteps

        out += ' ' * (indent + 2) + 'Number of utterances: %s\n' % len(self.vad_segments)
        out += segment_length_summary(self.vad_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        out += ' ' * (indent + 2) + 'Number of phones (characters): %s\n' % len(self.phn_segments)
        out += ' ' * (indent + 2) + 'Number of phone types: %s\n' % len(self.phn_segments.label.unique())
        out += segment_length_summary(self.phn_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        out += ' ' * (indent + 2) + 'Number of words: %s\n' % len(self.wrd_segments)
        out += ' ' * (indent + 2) + 'Number of word types: %s\n' % len(self.wrd_segments.label.unique())
        out += segment_length_summary(self.wrd_segments, indent=indent+2, steps_per_second=self.steps_per_second)

        return out



