import sys
import os
import math
import time
import numpy as np
import scipy.signal
import pandas as pd
from scipy.interpolate import Rbf
from .audio import wav_to_mfcc


def binary2integer_np(b, int_type='int32'):
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


def smooth_segmentations(segs, smoothing_factor=1., function='multiquadric', offset=10, n_points=None):
    n_frames = len(segs)
    max_len = n_frames * float(offset) / 1000
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
        offset=10,
        n_points=None,
        return_plot=False
):
    assert not (algorithm is None and n_points), 'Extraction at custom timepoints (n_points != None) is not supported when smoothing is turned of (algorithm==None).'
    n_frames = len(segs)
    max_len = n_frames * float(offset) / 1000
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
            timestamps = seg_ix * float(offset) / 1000 + float(offset) / (1000 * 2)
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
        offset=10,
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
            offset=offset,
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
        offset=10,
        activation='tanh',
        discretize=True,
        as_categories=True
):
    assert activation in ['sigmoid', 'softmax', 'tanh', 'linear', None]
    assert not (activation in ['linear', None] and discretize), "States with linear activation can't be discretized."

    ix = np.minimum(np.floor((timestamps * 1000) / offset), len(states) - 1).astype('int')
    out = states[ix]

    if discretize:
        if activation in ['sigmoid', 'tanh']:
            if activation == 'tanh':
                out = (out + 1) / 2
            out = np.rint(out).astype('int')

            if as_categories:
                out = binary2integer_np(out)

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
        offset=10,
        activation='tanh',
        discretize=True,
        as_categories=True,
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
            offset=offset,
            activation=activation,
            discretize=discretize,
            as_categories=as_categories
        )

        out.append(out_cur)

    return out


def binary_segments_to_intervals_inner(binary_segments, mask, src_segments=None, labels=None, offset=10):
    assert binary_segments.shape == mask.shape, 'binary_segments and mask must have the same shape'

    seg_indices1, seg_indices2 = np.where(binary_segments * mask)

    if src_segments is None:
        timestamps_ends = (np.reshape(mask.cumsum(), mask.shape)) / 1000 * offset
        ends = timestamps_ends[seg_indices1, seg_indices2]
        out = pd.DataFrame({'end': ends})
        out['start'] = out.end.shift().fillna(0.)
    else:
        timestamps_ends = mask.cumsum(axis=1) / 1000 * offset
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


def binary_segments_to_intervals(binary_segments, mask, file_indices, src_segments=None, labels=None, offset=10):
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
            offset=offset
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


def segment_length_summary(segs, indent=0, offset=None):
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

    if offset:
        out += ' ' * (indent + 2) + 'Frames:\n'
        out += ' ' * (indent + 4) + 'Min: %.4f\n' % (seg_min / offset * 1000)
        out += ' ' * (indent + 4) + 'Max: %.4f\n' % (seg_max / offset * 1000)
        out += ' ' * (indent + 4) + 'Max: %.4f\n' % (seg_max / offset * 1000)
        out += ' ' * (indent + 4) + '95th percentile: %.4f\n' % (seg_95 / offset * 1000)
        out += ' ' * (indent + 4) + '99th percentile: %.4f\n' % (seg_99 / offset * 1000)
        out += ' ' * (indent + 4) + '99.9th percentile: %.4f\n' % (seg_999 / offset * 1000)
        out += ' ' * (indent + 4) + 'Mean: %.4f\n' % (seg_mean / offset * 1000)

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


class AcousticDataset(object):
    def __init__(
            self,
            dir_path,
            sr = 16000,
            offset = 10,
            window_len = 25,
            n_coef = 13,
            order = 2,
            clip_timesteps=None,
            verbose=True
    ):
        assert sr % 1000 == 0, 'Must use a sampling rate that is a multiple of 1000'

        self.dir_path = dir_path
        self.sr = sr
        self.offset = offset
        self.window_len = window_len
        self.n_coef = n_coef
        self.order = order
        self.clip_timesteps = clip_timesteps

        self.data = {}

        wav_files = ['%s/%s' %(self.dir_path, x) for x in os.listdir(self.dir_path) if (x.endswith('.wav') or x.endswith('.WAV'))]
        n = len(wav_files)

        times = []
        for i, wav_file in enumerate(wav_files):
            t0 = time.time()
            if verbose:
                file_name = wav_file.split('/')[-1]
                if len(file_name) > 10:
                    file_name = file_name[:7] + '...'
                out_str = '\rProcessing file "%s" %d/%d' %(file_name, i + 1, n)
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

            new_data = AcousticDatafile(
                wav_file,
                sr=sr,
                offset=offset,
                window_len=window_len,
                n_coef=n_coef,
                order=order,
                clip_timesteps=clip_timesteps
            )

            self.data[new_data.ID] = new_data
            t1 = time.time()
            times.append(t1-t0)
            mean_time = sum(times) / len(times)
            eta = (n - i + 1) * mean_time

        if verbose:
            sys.stderr.write('\n')

        self.fileIDs = sorted(list(self.data.keys()))
        self.built = False

        self.vad_perm = None
        self.vad_perm_inv = None
        self.wrd_perm = None
        self.wrd_perm_inv = None
        self.phn_perm = None
        self.phn_perm_inv = None

    def features(self, fold=None, filter=None):
        out = []
        new_series = []
        for f in self.fileIDs:
            if filter:
                feats, _ = self.data[f].segment_and_stack(
                    segments=filter,
                    padding=None
                )
                feats = np.concatenate(
                    feats,
                    axis=1
                )
            else:
                feats = self.data[f].feats[None, ...]
            if fold:
                to_add = []
                for i in range(0, feats.shape[1], fold):
                    if i == 0:
                        new_series.append(1)
                    else:
                        new_series.append(0)
                    to_add.append(feats[:, i:i+fold])
                out += to_add
            else:
                new_series.append(1)
                out.append(feats)
                
        new_series = np.array(new_series)

        return out, new_series

    def segment_and_stack(
            self,
            feat_type='acoustic',
            segments='vad',
            max_len=None,
            padding='pre',
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
                    feat_type=feat_type,
                    segments=segments_cur,
                    max_len=max_len,
                    padding=padding,
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
        return self.segment_and_stack(
            feat_type=inner_segments,
            segments=outer_segments,
            max_len=max_len,
            padding=padding,
            reverse=reverse
        )

    def segments(self, segment_type='vad'):
        return pd.concat([self.data[f].segments(segment_type=segment_type) for f in self.fileIDs], axis=0)

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
            segmentation_probs,
            parent_segment_type='vad',
            states=None,
            discretize=True,
            state_activation='tanh',
            algorithm=None,
            algorithm_params=None,
            offset=10,
            n_points=None,
            mask=None,
            batch_mask=None,
            padding=None
    ):
        n_levels = len(segmentation_probs)
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
                [s[i:i+n_utt] for s in segmentation_probs],
                parent_segment_type=parent_segment_type,
                states=[s[i:i+n_utt] for s in states],
                discretize=discretize,
                mask=mask[i:i + n_utt],
                state_activation=state_activation,
                algorithm=algorithm,
                algorithm_params=algorithm_params,
                offset=offset,
                n_points=n_points,
                padding=padding
            )

            for j in range(n_levels):
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
                pred_cur = pred[f]

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


    def dump_segmentations_to_textgrid(self, outdir=None, suffix='', segments=None):
        for f in self.fileIDs:
            if isinstance(segments, str):
                segments_cur = segments
            else:
                segments_cur = segments[f]

            self.data[f].dump_segmentations_to_textgrid(
                outdir=outdir,
                suffix=suffix,
                segments=segments_cur
            )

    def summary(self, indent=0, summarize_components=False):
        out = ' ' * indent + 'DATASET SUMMARY:\n\n'
        out += ' ' * (indent + 2) + 'Source directory: %s\n' %self.dir_path
        duration = sum([self.data[f].duration for f in self.data])
        length = sum([len(self.data[f]) for f in self.data])
        out += ' ' * (indent + 2) + 'Total duration (seconds): %.4f\n' %duration
        out += ' ' * (indent + 2) + 'Total length (frames): %.4f\n' %length
        out += ' ' * (indent + 2) + 'Sampling rate: %s\n' %self.sr
        out += ' ' * (indent + 2) + 'Frame length: %sms\n' %self.window_len
        out += ' ' * (indent + 2) + 'Step size: %sms\n' %self.offset
        out += ' ' * (indent + 2) + 'Number of cepstral coefficients: %s\n' %self.n_coef
        out += ' ' * (indent + 2) + 'Number of derivatives: %s\n' %self.order

        if self.clip_timesteps:
            out += ' ' * (indent + 2) + 'Frame clipping limit: %s\n' %self.clip_timesteps

        out += '\n'
        vad_segments = self.segments('vad')
        out += ' ' * (indent + 2) + 'Number of VAD segments: %s\n' % len(vad_segments)
        out += segment_length_summary(vad_segments, indent=indent+2, offset=self.offset)

        out += '\n'
        phn_segments = self.segments('phn')
        out += ' ' * (indent + 2) + 'Number of phone segments: %s\n' % len(phn_segments)
        out += ' ' * (indent + 2) + 'Number of phone types: %s\n' % len(phn_segments.label.unique())
        out += segment_length_summary(phn_segments, indent=indent+2, offset=self.offset)

        out += '\n'
        wrd_segments = self.segments('wrd')
        out += ' ' * (indent + 2) + 'Number of word segments: %s\n' % len(wrd_segments)
        out += ' ' * (indent + 2) + 'Number of word types: %s\n' % len(wrd_segments.label.unique())
        out += segment_length_summary(wrd_segments, indent=indent+2, offset=self.offset)

        if summarize_components:
            out += '\n' + ' ' * (indent + 2) + 'COMPONENT DATA FILE SUMMARIES:\n\n'
            for f in self.data:
                out += self.data[f].summary(indent=indent+4, report_metadata=False)
                out += '\n'

        return out




class AcousticDatafile(object):
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
        assert path.endswith('.wav') or path.endswith('.WAV'), 'Input file "%s" was not .wav.' %path
        assert sr % 1000 == 0, 'Must use a sampling rate that is a multiple of 1000'

        self.sr = sr
        self.offset = offset
        self.window_len = window_len
        self.n_coef = n_coef
        self.order = order

        self.clip_timesteps = clip_timesteps

        feats, duration = wav_to_mfcc(
            path,
            sr=sr,
            offset=offset,
            window_len=window_len,
            n_coef=n_coef,
            order=order
        )
        feats = np.transpose(feats, [1, 0])

        if self.clip_timesteps is not None:
            feats = feats[:clip_timesteps, :]

        self.feats = feats
        self.shape = self.feats.shape

        self.len = feats.shape[0]
        self.duration = duration

        self.ID = os.path.basename(path)[:-4]
        self.dir = os.path.dirname(path)
        self.wav_path = path

        vad_path = self.dir + '/' + self.ID + '.vad'
        if os.path.exists(vad_path):
            self.vad_path = vad_path
            self.vad_segments = pd.read_csv(self.vad_path, sep=' ')
            self.vad_segments.speaker = self.vad_segments.speaker.astype('str')
            self.vad_segments.sort_values('start', inplace=True)
            self.vad_segments['label'] = 0
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

    def __len__(self):
        return self.len

    def segments(self, segment_type='vad'):
        if segment_type == 'vad':
            return self.vad_segments
        if segment_type == 'wrd':
            return self.wrd_segments
        if segment_type == 'phn':
            return self.phn_segments
        if segment_type == 'rnd':
            return self.rnd_segments

    def segments_to_one_hot(self, segments):
        one_hot = np.zeros(len(self))
        ix = np.array(segments.end)
        ix *= 1000 / self.offset
        ix = ix.astype('int')
        one_hot[ix] = 1

        return one_hot

    def one_hot_boundaries(self, segment_type='vad'):
        segments = self.segments(segment_type)
        one_hot = self.segments_to_one_hot(segments)

        return one_hot

    def segment_and_stack(
            self,
            feat_type='acoustic',
            segments='vad',
            max_len=None,
            padding='pre',
            reverse=False,
            normalize=False,
            center=False,
            with_deltas=True,
            resample=None
    ):
        if not isinstance(feat_type, list):
            feat_type = [feat_type]

        feats = []

        for f in feat_type:
            if f.lower() == 'acoustic':
                feats.append(self.feats)
            else :
                feats.append(self.one_hot_boundaries(segment_type=f)[...,None])

        feats = np.concatenate(feats, axis=-1)

        assert not (normalize and center), 'normalize and center cannot both be true, since normalize constrains to the interval [0,1] and center recenters at 0.'
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

        bounds_left = np.floor(segments_arr.start * 1000 / self.offset)
        bounds_right = np.ceil(segments_arr.end * 1000 / self.offset)
        bounds = np.stack([bounds_left, bounds_right], axis=1).astype(np.int32)

        feats_split = []
        mask_split = []
        for i in range(len(bounds)):
            s = bounds[i, 0]
            e = bounds[i, 1]
            if max_len is not None:
                e = min(e, s + max_len)
            new_feats = feats[s:e]

            if not ('acoustic' in [f.lower() for f in feat_type]):
                # Returning 1-hot segments, not acoustics, insert final boundary
                new_feats[-1] = 1

            length = e - s
            if resample and new_feats.shape[0] > 0:
                length = resample
                new_feats = scipy.signal.resample(new_feats, resample, axis=0)

            if not with_deltas:
                new_feats = new_feats[:,:self.n_coef]

            if normalize:
                maximum = new_feats.max()
                minimum = new_feats.min()
                diff = maximum - minimum
                new_feats = (new_feats - minimum) / diff
            if center:
                new_feats = new_feats - new_feats.mean()

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
                e = min(e + np.random.uniform(0., mean_frames_per_segment * self.offset / 1000 * 2), e_p)
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
            segmentation_probs,
            parent_segment_type='vad',
            states=None,
            discretize=True,
            state_activation='tanh',
            algorithm=None,
            algorithm_params=None,
            offset=10,
            n_points=None,
            mask=None,
            padding=None,
            snap_ends=True
    ):
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

        n_layers = len(segmentation_probs)

        for i in range(n_layers):
            starts = []
            ends = []
            labels = []

            timestamps = extract_segment_timestamps_batch(
                segmentation_probs[i],
                algorithm=algorithm,
                algorithm_params=algorithm_params,
                offset=offset,
                n_points=n_points,
                mask=mask,
                padding=padding
            )

            if states is not None:
                states_extracted = extract_states_at_timestamps_batch(
                    timestamps,
                    states[i],
                    offset=offset,
                    activation=state_activation,
                    discretize=discretize,
                    as_categories=True,
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
        start = np.where((np.arange(len(start))) * start)[0] * self.offset / 1000

        end[-1] = 1
        if mask is not None:
            end = (end + mask_ends) * mask
        end = (np.where((np.arange(len(end))) * end)[0] + 1) * self.offset / 1000

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
            ix = np.floor(segs.start * 1000 / self.offset).astype('int')
        else:
            ix = np.floor(segs.end * 1000 / self.offset).astype('int') - 1

        out[ix] = 1.

        return out

    def segs2mask(self, segs, max_len=None):
        out = np.zeros(len(self))

        bounds_left = np.floor(segs.start * 1000 / self.offset)
        bounds_right = np.ceil(segs.end * 1000 / self.offset)
        bounds = np.stack([bounds_left, bounds_right], axis=1).astype(np.int32)

        for i in range(len(bounds)):
            s = bounds[i, 0]
            e = bounds[i, 1]
            if max_len is not None:
                e = min(e, s + max_len)
            out[s:e] = 1

        return out

    def dump_segmentations_to_textgrid(self, outdir=None, suffix='', segments=None):
        if outdir is None:
            outdir = self.dir

        if isinstance(segments, str):
            segment_type = segments
            segments = self.segments(segment_type)
        else:
            segment_type = 'pred'

        path = outdir + '/' + self.ID + '_' + segment_type + suffix + '.TextGrid'
        with open(path, 'w') as f:
            f.write('File type = "ooTextFile"\n')
            f.write('Object class = "TextGrid"\n')
            f.write('\n')
            f.write('xmin = %.3f\n' % segments.start.iloc[0])
            f.write('xmax = %.3f\n' % segments.end.iloc[-1])
            f.write('tiers? <exists>\n')
            f.write('size = 1\n')
            f.write('item []:\n')
            f.write('    item [1]:\n')
            f.write('        class = "IntervalTier"\n')
            f.write('        class = "segmentations"\n')
            f.write('        xmin = %.3f\n' % segments.start.iloc[0])
            f.write('        xmax = %.3f\n' % segments.end.iloc[-1])
            f.write('        intervals: size = %d\n' % len(segments))

            row_str = '        intervals [%d]:\n' + \
                '            xmin = %.3f\n' + \
                '            xmax = %.3f\n' + \
                '            text = "%s"\n\n'

            if 'label' in segments.columns:
                for i, r in segments[['start', 'end', 'label']].iterrows():
                    f.write(row_str % (i, r.start, r.end, r.label))
            else:
                for i, r in segments[['start', 'end']].iterrows():
                    f.write(row_str % (i, r.start, r.end, ''))

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


    def summary(self, indent=0, report_metadata=True):
        out = ' ' * indent + 'DATA FILE SUMMARY: %s\n\n' %self.ID
        out += ' ' * (indent + 2) + 'Source location: %s\n' %self.wav_path
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
        out += segment_length_summary(self.vad_segments, indent=indent+2, offset=self.offset)

        if self.phn_path:
            out += '\n' + ' ' * (indent + 2) + 'Phone segmentation file location: %s\n' %self.phn_path
        else:
            out += '\n' + ' ' * (indent + 2) + 'No phone segmentation file provided.\n'
        out += ' ' * (indent + 2) + 'Number of phone segments: %s\n' % len(self.phn_segments)
        out += ' ' * (indent + 2) + 'Number of phone types: %s\n' % len(self.phn_segments.label.unique())
        out += segment_length_summary(self.phn_segments, indent=indent+2, offset=self.offset)

        if self.wrd_path:
            out += '\n' + ' ' * (indent + 2) + 'Word segmentation file location: %s\n' %self.wrd_path
        else:
            out += '\n' + ' ' * (indent + 2) + 'No word segmentation file provided.\n'
        out += ' ' * (indent + 2) + 'Number of word segments: %s\n' % len(self.wrd_segments)
        out += ' ' * (indent + 2) + 'Number of word types: %s\n' % len(self.wrd_segments.label.unique())
        out += segment_length_summary(self.wrd_segments, indent=indent+2, offset=self.offset)

        return out




