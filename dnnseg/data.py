import sys
import os
import time
import numpy as np
import pandas as pd
from .audio import wav_to_mfcc

def pad_sequence(sequence, seq_shape=None, dtype='float32', padding='pre', value=0.):
    assert padding in ['pre', 'post'], 'Padding type "%s" not recognized' % padding
    if seq_shape is None:
        seq_shape = get_seq_shape(sequence)

    if len(seq_shape) == 0:
        return sequence
    if isinstance(sequence, list):
        sequence = np.array([pad_sequence(x, seq_shape=seq_shape[1:], dtype=dtype, padding=padding, value=value) for x in sequence])
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

class AcousticDataset(object):
    def __init__(
            self,
            dir_path,
            sr = 16000,
            offset = 10,
            window_len = 25,
            n_coef = 13,
            order = None,
            clip_timesteps=None,
            verbose=True
    ):
        assert sr % 1000 == 0, 'Must use a sampling rate that is a multiple of 1000'

        self.dir_path = dir_path
        self.sr = sr
        self.offset = offset
        self.window_len = window_len
        self.n_coef = n_coef
        if order is None:
            order = [1, 2]
        elif not isinstance(order, list):
            order = [order]
        self.order = sorted(list(set(order)))
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

    def segment_and_stack(self, segmentations='vad'):
        if segmentations not in ['vad', 'wrd', 'phn']:
            assert isinstance(segmentations, dict) and sorted(list(segmentations.keys())) == self.fileIDs, 'Segmentations must either be in ["vad", "wrd", "phn"] or be of type dict with exactly one entry for each file in the dataset'
        feats = []
        mask = []
        for f in self.fileIDs:
            if segmentations in ['vad', 'wrd', 'phn']:
                f_segmentations = segmentations
            else:
                f_segmentations = segmentations[f]
            new_feats, new_mask = self.data[f].segment_and_stack(segmentations=f_segmentations)
            feats.append(new_feats)
            mask.append(new_mask)
        max_len = 0
        for f in feats:
            max_len = max(max_len, f.shape[-2])
        for i, f in enumerate(feats):
            feats[i] = repad_acoustic_features(feats[i], max_len)
        for i, f in enumerate(mask):
            mask[i] = np.squeeze(repad_acoustic_features(mask[i][...,None], max_len), -1)
        return np.concatenate(feats, axis=0), np.concatenate(mask, axis=0)

    def inputs(self):
        return self.segment_and_stack(segmentations='vad')

    def words(self):
        return self.segment_and_stack(segmentations='wrd')

    def phones(self):
        return self.segment_and_stack(segmentations='phn')

    def targets(self, reverse=True, with_deltas=False, segment_type='vad'):
        targets, mask = self.segment_and_stack(segmentations=segment_type)
        if not with_deltas:
            targets = targets[...,:self.n_coef]
        if reverse:
            return targets[:, ::-1], mask[:, ::-1]
        return targets[:, :], mask

    def segments(self, segment_type='vad'):
        if segment_type == 'vad':
            return pd.concat([self.data[f].vad_segments for f in self.fileIDs], axis=0)
        if segment_type == 'wrd':
            return pd.concat([self.data[f].wrd_segments for f in self.fileIDs], axis=0)
        if segment_type == 'phn':
            return pd.concat([self.data[f].phn_segments for f in self.fileIDs], axis=0)

    def classes(self, segment_type='wrd'):
        assert segment_type in ['phn', 'wrd'], 'Only segment types "phn" and "wrd" are supported for label extraction'

        labels = self.segments(segment_type=segment_type).label
        ix2label = sorted(list(labels.unique()))

        return ix2label

    def n_classes(self, segment_type='wrd'):
        assert segment_type in ['phn', 'wrd'], 'Only segment types "phn" and "wrd" are supported for label extraction'

        return len(self.classes(segment_type=segment_type))

    def labels(self, one_hot=True, segment_type='wrd'):
        assert segment_type in ['phn', 'wrd'], 'Only segment types "phn" and "wrd" are supported for label extraction'

        labels = self.segments(segment_type=segment_type).label
        ix2label = self.classes(segment_type=segment_type)

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




class AcousticDatafile(object):
    def __init__(
            self,
            path,
            sr = 16000,
            offset = 10,
            window_len = 25,
            n_coef = 13,
            order = None,
            clip_timesteps=None
    ):
        assert path.endswith('.wav') or path.endswith('.WAV'), 'Input file "%s" was not .wav.' %path
        assert sr % 1000 == 0, 'Must use a sampling rate that is a multiple of 1000'

        self.sr = sr
        self.offset = offset
        self.window_len = window_len
        self.n_coef = n_coef

        if order is None:
            order = [1, 2]
        elif not isinstance(order, list):
            order = [order]
        self.order = sorted(list(set(order)))

        self.clip_timesteps = clip_timesteps

        feats = wav_to_mfcc(
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

        self.ID = os.path.basename(path)[:-4]
        self.dir = os.path.dirname(path)
        self.wav_path = path

        vad_path = self.dir + '/' + self.ID + '.vad'
        if os.path.exists(vad_path):
            self.vad_path = vad_path
            self.vad_segments = pd.read_csv(self.vad_path, sep=' ')
        else:
            self.vad_path = None
            self.vad_segments = pd.DataFrame(
                {
                    'start': [0.],
                    'end': [float(self.len)],
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

    def __len__(self):
        return self.len

    def segment_and_stack(self, segmentations='vad'):
        if segmentations == 'vad':
            segmentations = self.vad_segments
        elif segmentations == 'wrd':
            segmentations = self.wrd_segments
        elif segmentations == 'phn':
            segmentations = self.phn_segments

        bounds = np.array(
            np.rint((segmentations[['start', 'end']] * 100)),
            dtype=np.int32
        )
        feats_split = []
        for i in range(len(bounds)):
            feats_split.append(self.feats[bounds[i, 0]:bounds[i, 1], :])
        feats = pad_sequence(feats_split)
        mask = np.all(np.logical_not(np.isclose(feats, 0.)), axis=2)

        return feats, mask



