import sys
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib
from matplotlib import ticker, pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy

from .data import extract_segment_timestamps

class SmartDict(dict):
    def __missing__(self, key):
        return key


def plot_streaming_segmenter(
        input,
        segmentation,
        left_targ,
        left_pred,
        right_targ,
        right_pred,
        cmap='Blues',
        titles=None,
        dir='./',
        prefix='',
        suffix='.png'
):
    pass



def plot_acoustic_features(
        inputs,
        targets,
        preds,
        segmentation_probs=None,
        states=None,
        target_means=None,
        sr=16000,
        hop_length=160,
        drop_zeros=True,
        hard_segmentations=False,
        cmap='Blues',
        titles=None,
        label_map=None,
        dir='./',
        prefix='',
        suffix='.png'
):
    time_tick_formatter = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * hop_length/sr))

    if titles is None:
        titles = [None] * inputs.shape[0]

    if label_map is not None:
        label_map = dict(zip(label_map.source,label_map.target))

    has_means = target_means is not None
    has_segs = segmentation_probs is not None
    has_states = states is not None
    n_states = len(states) if has_states else 0

    fig = plt.figure()

    n_plots = 3
    if has_means:
        n_plots += 1
    if has_segs:
        n_plots += 2
        if not hard_segmentations:
            n_plots += 1
    if has_states:
        n_plots += n_states

    fig.set_size_inches(10, 3 * n_plots)
    axes = []
    for i in range(n_plots):
        axes.append(fig.add_subplot(n_plots, 1, i+1))

    segmentations_offset = 0
    target_means_offset = 0
    states_offset = 0

    ax_input = axes[0]

    if has_segs:
        ax_seg = axes[1]
        ax_seg_hard = axes[2]
        segmentations_offset = 2
        if not hard_segmentations:
            ax_seg_smoothed = axes[3]
            segmentations_offset += 1

    if has_means:
        ax_targ_means = axes[1 + segmentations_offset]
        target_means_offset = 1

    if has_states:
        ax_states = []
        for i in range(n_states):
            ax_states.append(axes[1 + segmentations_offset + target_means_offset + i])
        states_offset = n_states

    ax_targ = axes[1 + segmentations_offset + target_means_offset + states_offset]
    ax_pred = axes[2 + segmentations_offset + target_means_offset + states_offset]

    for i in range(len(inputs)):
        inputs_cur = inputs[i]
        if drop_zeros:
            inputs_select = np.where(np.all(np.logical_not(np.isclose(inputs_cur, 0.)), axis=1))[0]
            inputs_cur = inputs_cur[inputs_select]
        inputs_cur = np.swapaxes(inputs_cur, -2, -1)

        librosa.display.specshow(
            inputs_cur,
            sr=sr,
            hop_length=hop_length,
            fmax=8000,
            x_axis='time',
            cmap=cmap,
            ax=ax_input
        )
        ax_input.set_title('Inputs')

        if has_segs:
            segmentation_probs_cur = segmentation_probs[i]
            if drop_zeros:
                segmentation_probs_cur = segmentation_probs_cur[inputs_select]

            segmentation_probs_cur = np.swapaxes(segmentation_probs_cur, -2, -1)

            df = pd.DataFrame(
                segmentation_probs_cur,
                index=list(range(1, segmentation_probs_cur.shape[0] + 1))
            )

            ax_seg.pcolormesh(df, cmap='Greys', vmin=0., vmax=1.)
            ax_seg.xaxis.set_major_formatter(time_tick_formatter)
            ax_seg.set_yticks(np.arange(0.5, len(df.index), 1), minor=False)
            ax_seg.set_yticklabels(df.index)
            ax_seg.set_xlabel('Time')
            ax_seg.set_title('Segmentations')

            colors = [plt.get_cmap('gist_rainbow')(1. * j / len(segmentation_probs_cur)) for j in range(len(segmentation_probs_cur))]
            segs_hard = []

            if hard_segmentations:
                smoothing_algorithm = None
                n_points = None
            else:
                smoothing_algorithm = 'rbf'
                n_points = 1000

            for j, s in enumerate(segmentation_probs_cur):
                timestamps, basis, segs_smoothed = extract_segment_timestamps(
                    s,
                    algorithm=smoothing_algorithm,
                    n_points=n_points,
                    return_plot=True
                )
                segs_hard.append(timestamps)
                if not hard_segmentations:
                    ax_seg_smoothed.plot(basis, segs_smoothed, color=colors[j], label=str(j+1))
                ax_seg_hard.vlines(timestamps, j, j+1, color='k')

            if not hard_segmentations:
                ax_seg_smoothed.set_xlim(0., basis[-1])
                ax_seg_smoothed.legend(fancybox=True, framealpha=0.75, frameon=True, facecolor='white', edgecolor='gray')
                ax_seg_smoothed.set_xlabel('Time')
                ax_seg_smoothed.set_title('Smoothed Segmentations')

            ax_seg_hard.set_xlim(0., basis[-1])
            ax_seg_hard.set_ylim(0., n_states - 1)
            ax_seg_hard.set_yticks(np.arange(0.5, len(df.index), 1), minor=False)
            ax_seg_hard.set_yticklabels(df.index)
            ax_seg_hard.set_xlabel('Time')
            ax_seg_hard.set_title('Hard segmentations')

        if has_states:
            for j in range(n_states):
                states_cur = states[j][i]

                if drop_zeros:
                    states_cur = states_cur[inputs_select]

                states_cur = np.swapaxes(states_cur, -2, -1)

                librosa.display.specshow(
                    states_cur,
                    sr=sr,
                    hop_length=hop_length,
                    fmax=8000,
                    x_axis='time',
                    cmap=cmap,
                    ax=ax_states[j]
                )
                ax_states[j].set_title('Hidden States (%s)' %(j+1))

        if has_means:
            target_means_cur = target_means[i]
            if drop_zeros:
                target_means_select = np.where(np.all(np.logical_not(np.isclose(target_means_cur, 0.)), axis=1))[0]
                target_means_cur = target_means_cur[target_means_select]
            target_means_cur = np.swapaxes(target_means_cur, -2, -1)

            librosa.display.specshow(
                target_means_cur,
                sr=sr,
                hop_length=hop_length,
                fmax=8000,
                x_axis='time',
                cmap=cmap,
                ax=ax_targ_means
            )
            ax_targ_means.set_title('Target means')

        targets_cur = targets[i]
        if drop_zeros:
            targets_select = np.where(np.all(np.logical_not(np.isclose(targets_cur, 0.)), axis=1))[0]
            targets_cur = targets_cur[targets_select]
        targets_cur = np.swapaxes(targets_cur, -2, -1)

        librosa.display.specshow(
            targets_cur,
            sr=sr,
            hop_length=hop_length,
            fmax=8000,
            x_axis='time',
            cmap=cmap,
            ax=ax_targ
        )
        ax_targ.set_title('Targets')

        preds_cur = preds[i]
        if drop_zeros:
            preds_select = np.where(np.all(np.logical_not(np.isclose(preds_cur, 0.)), axis=1))[0]
            preds_cur = preds_cur[preds_select]
        preds_cur = np.swapaxes(preds_cur, -2, -1)

        librosa.display.specshow(
            preds_cur,
            sr=sr,
            hop_length=hop_length,
            fmax=8000,
            x_axis='time',
            cmap=cmap,
            ax=ax_pred
        )
        ax_pred.set_title('Predictions')

        if titles[i] is not None:
            title = titles[i]
            if label_map is not None:
                title = label_map.get(title, title)
            fig.suptitle(title, fontsize=20, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            fig.savefig(dir + '/' + prefix + 'featureplot_%d'%i + suffix)
        except:
            sys.stderr.write('IO error when saving plot. Skipping plotting...\n')

        ax_input.clear()
        if has_segs:
            ax_seg.clear()
            ax_seg_hard.clear()
            if not hard_segmentations:
                ax_seg_smoothed.clear()
        if has_states:
            for ax in ax_states:
                ax.clear()
        if has_means:
            ax_targ_means.clear()
        ax_targ.clear()
        ax_pred.clear()

    plt.close(fig)


def plot_label_histogram(labels, title=None, bins='auto', label_map=None, dir='./', prefix='', suffix='.png'):

    if label_map is not None:
        label_map = dict(zip(label_map.source,label_map.target))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    labels = pd.Series(labels)
    if label_map is not None:
        labels = labels.replace(label_map)

    ax.hist(labels, bins=bins)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    try:
        fig.savefig(dir + '/' + prefix + 'label_histogram' + suffix)
    except:
        sys.stderr.write('IO error when saving plot. Skipping plotting...\n')

    plt.close(fig)

def plot_label_heatmap(labels, preds, title=None, label_map=None, cmap='Blues', dir='./', prefix='', suffix='.png'):
    if label_map is not None:
        label_map = dict(zip(label_map.source,label_map.target))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    df = pd.crosstab(labels, preds)

    df = pd.DataFrame(np.array(df) / np.array(df).sum(axis=0, keepdims=True), index=df.index, columns=df.columns)
    if label_map is not None:
        index = pd.Series(df.index).replace(label_map)
    else:
        index = df.index
    df.index = index
    df.index.name = None

    fig.set_size_inches(1 + .5 * len(df.columns), .27 * len(df.index))

    ax.pcolormesh(df, cmap=cmap, vmin=0., vmax=1.)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1), minor=False)
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(0.5, len(df.index), 1), minor=False)
    ax.set_yticklabels(df.index)

    if title is not None:
        fig.suptitle(title)
    # fig.tight_layout()

    try:
        fig.savefig(dir + '/' + prefix + 'label_heatmap' + suffix)
    except:
        sys.stderr.write('IO error when saving plot. Skipping plotting...\n')

    plt.close(fig)

def plot_binary_unit_heatmap(labels, label_probs, title=None, label_map=None, cmap='Blues', dir='./', prefix='', suffix='.png', dump_data=True):
    if label_map is not None:
        label_map = dict(zip(label_map.source,label_map.target))

    df = {'label': labels}
    for bit in range(label_probs.shape[1]):
        df['%d' %bit] = label_probs[:,bit]

    df = pd.DataFrame(df)
    df = df[~df['label'].isin(['SPN', 'SIL'])]
    df = df.groupby('label').mean()

    if label_map is not None:
        index = pd.Series(df.index).replace(label_map)
    else:
        index = df.index

    df.index = index
    df.index.name = None

    width = 1 + .3 * len(df.columns)
    height = .27 * len(df.index)

    # Correlation-based clustering is undefined if any vectors happen to have no variance.
    # Try correlation first, then fall back to default distance metric if it fails.
    try:
        cm = sns.clustermap(data=df, metric='correlation', cmap=cmap, figsize=(width,height))
    except:
        cm = sns.clustermap(data=df, cmap=cmap, figsize=(width,height))
    cm.cax.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)

    try:
        cm.savefig(dir + '/' + prefix + 'binary_unit_heatmap' + suffix)
    except:
        sys.stderr.write('IO error when saving plot. Skipping plotting...\n')

    if dump_data:
        df.to_csv(dir + '/' + prefix + 'binary_unit_heatmap_data.csv')

    plt.close('all')



