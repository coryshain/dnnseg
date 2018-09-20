import sys
import os
import math
import time
import numpy as np
import scipy.signal
import pandas as pd
from scipy.interpolate import Rbf
from scipy.signal import find_peaks_cwt
from .audio import wav_to_mfcc

def get_segmentation_smoother(times, segs, smoothing_factor=1., function='multiquadric'):
    out = Rbf(times, segs, smooth=smoothing_factor, function=function)

    return out

def smooth_segmentations(segs, smoothing_factor=1., function='multiquadric', offset=10, n_points=None):
    n_frames = len(segs)
    max_len = n_frames * float(offset) / 1000
    x = np.linspace(0, max_len, n_frames)
    y = segs
    if n_points is None:
        n_points = n_frames
    basis = np.linspace(0, max_len, n_points)

    out = get_segmentation_smoother(x, y, smoothing_factor=smoothing_factor, function=function, offset=offset)(basis)

    return basis, out

def extract_segmentation_indices_1D(segs, smooth_type=None, smoother_params=None, offset=10, n_points=None, return_plot=False):
    n_frames = len(segs)
    max_len = n_frames * float(offset) / 1000
    x = np.linspace(0, max_len, n_frames)
    y = segs
    if smoother_params is None:
        smoother_params = {}

    if smooth_type == 'rbf':
        smoother = get_segmentation_smoother(x, y, **smoother_params)
        smoothed_segs = smoother(x)

        seg_ix = extract_peak_ix(smoothed_segs)[0]

        out = seg_ix

        if return_plot:
            if n_points is None:
                n_points = n_frames
            basis = np.linspace(0, max_len, n_points)
            response = smoother(basis)
            out = (seg_ix, basis, response)

    return out

def extract_segmentation_indices_2D(segs, smooth_type=None, smoother_params=None, offset=10, n_points=None, return_plot=False, mask=None):
    ix = None
    if return_plot:
        basis = None
        response = np.zeros((len(segs), n_points))

    if mask is not None:
        sequence_lengths = np.sum(mask, axis=1)

    for i in range(len(segs)):
        extraction = extract_segmentation_indices_1D(
            segs[i],
            smooth_type=smooth_type,
            smoother_params=smoother_params,
            offset=offset,
            n_points=n_points,
            return_plot=return_plot
        )
        if return_plot:
            ix_cur, basis_cur, response_cur = extraction
        else:
            ix_cur = extraction

        if mask is not None:
            ix_cur += sequence_lengths[i] - 1

        if ix is None:
            ix = [[np.repeat(i, len(ix_cur))], [ix_cur]]
        else:
            ix[0].append(np.repeat(i, len(ix_cur)))
            ix[1].append(ix_cur)

        if return_plot:
            if basis is None:
                basis = basis_cur[None, ...]
            response[i] = response_cur

    ix = (np.concatenate(ix[0], axis=0), np.concatenate(ix[1], axis=0))

    if return_plot:
        out = ix, basis_cur, response_cur

def extract_peak_ix(v):
    before = np.full_like(v, np.inf)
    before[..., 1:] = v[..., :-1]
    after = np.full_like(v, np.inf)
    after[..., :-1] = v[..., 1:]

    ix = np.where(np.logical_and(v > before, v > after))

    return ix

def find_segmentation_peaks(segs, widths, wavelet=None, min_snr=1):
    out = find_peaks_cwt(segs, widths, wavelet=wavelet, min_snr=min_snr)

    return out

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

def score_segs(true, pred, tol=0.02):
    true_lab = np.array(true.label)
    pred_lab = np.array(pred.label)
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

    lexicon = []

    while i < len(true) and j < len(pred):
        l_true = true_lab[i]
        l_pred = pred_lab[j]
        s_true, e_true = true[i]
        s_pred, e_pred = pred[j]

        # If we are at the final true segment or if there's a jump in time to the next true segment,
        # check the end boundary, too
        check_end = (i == len(true) - 1) or (math.fabs(true[i+1, 0] - true[i, 1]) > 1e-5)

        s_hit = False
        e_hit = False

        if math.fabs(s_true-s_pred) <= tol:
            s_hit = True
        if math.fabs(e_true-e_pred) <= tol:
            e_hit = True
        if s_hit and e_hit:
            lexicon.append([l_true, l_pred])

        b_tp += s_hit
        b_tp += e_hit and check_end
        w_tp += s_hit and e_hit

        if s_hit:
            i += 1
            j += 1
            if not e_hit:
                w_fp += 1
                w_fn += 1

        elif s_true < s_pred:
            i += 1
            if not s_hit:
                b_fn += 1
            if not e_hit:
                w_fn += 1
        else:
            j += 1
            if not s_hit:
                b_fp += 1
            if not e_hit:
                w_fp += 1

        if check_end and not e_hit:
            if e_true < e_pred:
                b_fn += 1
            else:
                b_fp += 1

    out = {
        'b_tp': b_tp,
        'b_fp': b_fp,
        'b_fn': b_fn,
        'w_tp': w_tp,
        'w_fp': w_fp,
        'w_fn': w_fn,
        # 'lexicon': lexicon
    }

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

    def segment_and_stack(
            self,
            segmentation_type='vad',
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
            new_feats, new_mask = self.data[f].segment_and_stack(
                segmentation_type=segmentation_type,
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
            segmentation_type='vad',
            max_len=None,
            padding='pre',
            reverse=False,
            normalize=False,
            center=False,
            resample=None
    ):
        return self.segment_and_stack(
            segmentation_type=segmentation_type,
            max_len=max_len,
            padding=padding,
            reverse=reverse,
            normalize=normalize,
            center=center,
            resample = resample
        )

    def targets(
            self,
            segmentation_type='vad',
            padding='post',
            max_len=None,
            reverse=True,
            normalize=False,
            center=False,
            with_deltas=False,
            resample=None
    ):
        targets, mask = self.segment_and_stack(
            segmentation_type=segmentation_type,
            max_len=max_len,
            padding=padding,
            reverse=reverse,
            normalize=normalize,
            center=center,
            with_deltas=with_deltas,
            resample=resample
        )
        return targets, mask

    def segments(self, segment_type='vad'):
        if segment_type == 'vad':
            return pd.concat([self.data[f].vad_segments for f in self.fileIDs], axis=0)
        if segment_type == 'wrd':
            return pd.concat([self.data[f].wrd_segments for f in self.fileIDs], axis=0)
        if segment_type == 'phn':
            return pd.concat([self.data[f].phn_segments for f in self.fileIDs], axis=0)
        if segment_type == 'rnd':
            return pd.concat([self.data[f].rnd_segments for f in self.fileIDs], axis=0)

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

    def dump_segmentations_to_textgrid(self, out_dir=None, suffix='', segmentations=None, segmentation_type='phn'):
        for f in self.fileIDs:
            self.data[f].dump_segmentations_to_textgrid(
                out_dir=out_dir,
                suffix=suffix,
                segmentations=segmentations,
                segmentation_type=segmentation_type
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
            feats = feats[:, :clip_timesteps]

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
            self.vad_segments['label'] = 0
        else:
            self.vad_path = None
            self.vad_segments = pd.DataFrame(
                {
                    'start': [0.],
                    'end': [float(self.len)],
                    'label': 0,
                    'speaker': 0,
                    'index': 0
                }
            )
        self.vad_segments.fileID = self.ID

        phn_path = self.dir + '/' + self.ID + '.phn'
        if os.path.exists(phn_path):
            self.phn_path = phn_path
            self.phn_segments = pd.read_csv(self.phn_path, sep=' ')
        else:
            self.phn_path = None
            if self.vad_path is None:
                self.phn_segments = pd.DataFrame(
                    {
                        'start': [0.],
                        'end': [float(self.len)],
                        'label': 0,
                        'speaker': 0,
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

    def segment_and_stack(
            self,
            segmentation_type='vad',
            max_len=None,
            padding='pre',
            reverse=False,
            normalize=False,
            center=False,
            with_deltas=True,
            resample=None
    ):
        assert not (normalize and center), 'normalize and center cannot both be true, since normalize constrains to the interval [0,1] and center recenters at 0.'
        if segmentation_type == 'vad':
            segmentations_arr = self.vad_segments
        elif segmentation_type == 'wrd':
            segmentations_arr = self.wrd_segments
        elif segmentation_type == 'phn':
            segmentations_arr = self.phn_segments
        elif segmentation_type == 'rnd':
            segmentations_arr = self.rnd_segments

        pad_seqs = padding not in ['None', None]

        bounds = np.array(
            np.rint((segmentations_arr[['start', 'end']] * 1000 / self.offset)),
            dtype=np.int32
        )
        feats_split = []
        mask_split = []
        for i in range(len(bounds)):
            s = bounds[i, 0]
            e = bounds[i, 1]
            if max_len is not None:
                e = min(e, s + max_len)
            new_feats = self.feats[s:e, :]
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

    def generate_random_segmentation(self, mean_frames_per_segment):
        vad_indicator_start = self.segs2indicator(self.vad_segments, location='start')
        vad_indicator_end = self.segs2indicator(self.vad_segments, location='end')
        vad_mask = self.segs2mask(self.vad_segments)

        rate = 1. / mean_frames_per_segment

        out = (np.random.random(len(self)) > (1 - rate)).astype('float')

        out = self.indicator2segs(
            out,
            mask=vad_mask,
            mask_starts=vad_indicator_start,
            mask_ends=vad_indicator_end,
            location='start'
        )

        return out

    def initialize_random_segmentation(self, rate):
        self.rnd_segments = self.generate_random_segmentation(rate)

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
        start = np.where((np.arange(len(start))) * start)[0]  * self.offset / 1000

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
            ix = np.rint(segs.start * 1000 / self.offset).astype('int')
        else:
            ix = np.rint(segs.end * 1000 / self.offset).astype('int') - 1

        out[ix] = 1.

        return out

    def segs2mask(self, segs, max_len=None):
        out = np.zeros(len(self))

        bounds = np.array(
            np.rint((segs[['start', 'end']] * 1000 / self.offset)),
            dtype=np.int32
        )

        for i in range(len(bounds)):
            s = bounds[i, 0]
            e = bounds[i, 1]
            if max_len is not None:
                e = min(e, s + max_len)
            out[s:e] = 1

        return out

    def dump_segmentations_to_textgrid(self, out_dir=None, suffix='', segmentations=None, segmentation_type='phn'):
        if out_dir is None:
            out_dir = self.dir

        assert segmentation_type in ['vad', 'phn', 'wrd', 'rand', 'pred'], "Only the following segmentation types are permitted: ['vad', 'phn', 'wrd', 'rand', 'pred']"

        if segmentations is None:
            assert segmentation_type in ['vad', 'phn', 'wrd'], "If no segmentations are provided, segmentation type must be one of ['vad', 'phn', 'wrd']"
            if segmentation_type == 'vad':
                segmentations = self.vad_segments
            if segmentation_type == 'phn':
                segmentations = self.phn_segments
            if segmentation_type == 'wrd':
                segmentations = self.wrd_segments

        path = out_dir + '/' + self.ID + '_' + segmentation_type + suffix + '.TextGrid'
        with open(path, 'w') as f:
            f.write('File type = "ooTextFile"\n')
            f.write('Object class = "TextGrid"\n')
            f.write('\n')
            f.write('xmin = %.3f\n' % segmentations.start.iloc[0])
            f.write('xmax = %.3f\n' % segmentations.end.iloc[-1])
            f.write('tiers? <exists>\n')
            f.write('size = 1\n')
            f.write('item []:\n')
            f.write('    item [1]:\n')
            f.write('        class = "IntervalTier"\n')
            f.write('        class = "segmentations"\n')
            f.write('        xmin = %.3f\n' % segmentations.start.iloc[0])
            f.write('        xmax = %.3f\n' % segmentations.end.iloc[-1])
            f.write('        intervals: size = %d\n' % len(segmentations))

            row_str = '        intervals [%d]:\n' + \
                '            xmin = %.3f\n' + \
                '            xmax = %.3f\n' + \
                '            text = "%s"\n\n'

            if 'label' in segmentations.columns:
                for i, r in segmentations[['start', 'end', 'label']].iterrows():
                    f.write(row_str % (i, r.start, r.end, r.label))
            else:
                for i, r in segmentations[['start', 'end']].iterrows():
                    f.write(row_str % (i, r.start, r.end, ''))

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




