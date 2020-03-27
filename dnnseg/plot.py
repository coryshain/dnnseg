import sys
import re
import numpy as np
import scipy.cluster.hierarchy as spc
from scipy import sparse
import librosa
import librosa.display
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker, pyplot as plt
import seaborn as sns
from .data import extract_segment_timestamps
from .util import stderr


is_embedding_dimension = re.compile('d([0-9]+)')


def get_cmap(i):
    j = i % 6
    if j == 0:
        return 'Blues'
    if j == 1:
        return 'Greens'
    if j == 2:
        return 'Reds'
    if j == 3:
        return 'Purples'
    if j == 4:
        return 'Oranges'
    if j == 5:
        return 'Greys'


def plot_acoustic_features(
        inputs,
        states=None,
        segmentation_probs=None,
        segmentation_probs_smoothed=None,
        segmentations=None,
        targs=None,
        preds=None,
        attn=None,
        attn_keys=None,
        positional_encodings=None,
        plot_keys=None,
        sr=16000,
        hop_length=160,
        drop_zeros=False,
        titles=None,
        label_map=None,
        directory='./',
        prefix='',
        suffix='.png'
):
    time_tick_formatter = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * hop_length/sr))

    if titles is None:
        titles = [None] * inputs.shape[0]

    compute_keys = plot_keys is None

    # Add plot labels if plots are provided as lists
    if compute_keys:
        plot_keys = set()
    if targs is not None:
        if isinstance(targs, list):
            targs = {str(x):y for x, y in zip(range(len(targs)), targs)}
        if compute_keys:
            plot_keys = plot_keys.union(set(targs.keys()))
    if preds is not None:
        if isinstance(preds, list):
            preds = {str(x):y for x, y in zip(range(len(preds)), preds)}
        if compute_keys:
            plot_keys = plot_keys.union(set(preds.keys()))
    if attn is not None:
        if isinstance(attn, list):
            attn = {str(x):y for x, y in zip(range(len(attn)), attn)}
        if compute_keys:
            plot_keys = plot_keys.union(set(attn.keys()))
    if attn_keys is not None:
        if isinstance(attn_keys, list):
            attn_keys = {str(x):y for x, y in zip(range(len(attn_keys)), attn_keys)}
        if compute_keys:
            plot_keys = plot_keys.union(set(attn_keys.keys()))
    if positional_encodings is not None:
        if isinstance(positional_encodings, list):
            positional_encodings = {str(x):y for x, y in zip(range(len(positional_encodings)), positional_encodings)}
        if compute_keys:
            plot_keys = plot_keys.union(set(positional_encodings.keys()))

    if compute_keys:
        plot_keys = sorted(list(plot_keys))

    has_segs = segmentation_probs is not None
    has_smoothing = segmentations is None or segmentation_probs_smoothed is not None
    has_states = states is not None
    n_states = len(states) if has_states else 0

    fig = plt.figure()

    n_plots = 1
    if has_segs:
        n_plots += 1
        if has_smoothing:
            n_plots += 1
    if has_states:
        n_plots += n_states
    if targs is not None:
        i = 0
        for x in targs:
            if x in plot_keys:
                i += 1
        n_plots += i
    if preds is not None:
        i = 0
        for x in preds:
            if x in plot_keys:
                i += 1
        n_plots += i
    if attn is not None:
        i = 0
        for x in attn:
            if x in plot_keys:
                i += 1
        n_plots += i
    if attn_keys is not None:
        i = 0
        for x in attn_keys:
            if x in plot_keys:
                i += 1
        n_plots += i
    if positional_encodings is not None:
        i = 0
        for x in positional_encodings:
            if x in plot_keys:
                i += 1
        n_plots += i

    fig.set_size_inches(10, 3 * n_plots)
    axes_src = []
    for i in range(n_plots):
        axes_src.append(fig.add_subplot(n_plots, 1, i+1))

    axes = {}

    axes['input'] = axes_src[0]
    plot_ix = 1

    if has_segs:
        axes['seg'] = axes_src[plot_ix]
        plot_ix += 1
        if has_smoothing:
            axes['seg_smoothed'] = axes_src[plot_ix]
            plot_ix += 1

    if has_states:
        for i in range(n_states):
            axes['states_%d' % i] = axes_src[plot_ix]
            plot_ix += 1

    for k in plot_keys:
        if targs is not None and k in targs:
            axes['Targets ' + k] = axes_src[plot_ix]
            plot_ix += 1
        if preds is not None and k in preds:
            axes['Predictions ' + k] = axes_src[plot_ix]
            plot_ix += 1
        if attn is not None and k in attn:
            axes['Top-Down Attention ' + k] = axes_src[plot_ix]
            plot_ix += 1
        if attn_keys is not None and k in attn_keys:
            axes['Top-Down Attention Keys ' + k] = axes_src[plot_ix]
            plot_ix += 1
        if positional_encodings is not None and k in positional_encodings:
            axes['Positional Encodings ' + k] = axes_src[plot_ix]
            plot_ix += 1

    n_feats = None

    for i in range(len(inputs)):
        ax = axes['input']
        inputs_cur = inputs[i]
        if n_feats is None:
            n_feats = inputs_cur.shape[-1]
        if drop_zeros:
            inputs_select = np.where(np.any(inputs_cur[:,:-1] != 0., axis=1))[0]
            inputs_cur = inputs_cur[inputs_select]
        inputs_cur = np.swapaxes(inputs_cur, -2, -1)

        librosa.display.specshow(
            inputs_cur,
            sr=sr,
            hop_length=hop_length,
            fmax=8000,
            x_axis='time',
            cmap=get_cmap(0),
            ax=ax
        )
        ax.set_title('Inputs')

        if has_segs:
            ax = axes['seg']
            segmentation_probs_cur = segmentation_probs[i]
            if drop_zeros:
                segmentation_probs_cur = segmentation_probs_cur[inputs_select]

            segmentation_probs_cur = np.swapaxes(segmentation_probs_cur, -2, -1)

            if segmentations is None:
                segmentations_cur = segmentation_probs_cur
            else:
                segmentations_cur = segmentations[i]
                if drop_zeros:
                    segmentations_cur = segmentations_cur[inputs_select]
                segmentations_cur = np.swapaxes(segmentations_cur, -2, -1)

            df = pd.DataFrame(
                segmentation_probs_cur,
                index=list(range(1, segmentation_probs_cur.shape[0] + 1))
            )

            pcm = axes['seg'].pcolormesh(df, cmap='Greys', vmin=0., vmax=1.)
            ax.xaxis.set_major_formatter(time_tick_formatter)
            ax.set_yticks(np.arange(0.5, len(df.index), 1), minor=False)
            ax.set_yticklabels(df.index)
            ax.set_xlabel('Time')
            ax.set_title('Segmentation Probabilities')

            if has_smoothing:
                ax_s = axes['seg_smoothed']
            colors = [plt.get_cmap('gist_rainbow')(1. * j / len(segmentation_probs_cur)) for j in range(len(segmentation_probs_cur))]

            if segmentations is None:
                smoothing_algorithm = 'rbf'
                n_points = 1000
            else:
                smoothing_algorithm = None
                n_points = None

            for j, s in enumerate(segmentations_cur):
                timestamps, basis, segs_smoothed = extract_segment_timestamps(
                    s,
                    algorithm=smoothing_algorithm,
                    n_points=n_points,
                    seconds_per_step=float(hop_length)/sr,
                    return_plot=True
                )
                if segmentation_probs_smoothed is not None:
                    segs_smoothed = segmentation_probs_smoothed[i][:,j]
                    if drop_zeros:
                        segs_smoothed = segs_smoothed[inputs_select]
                if has_smoothing:
                    ax_s.plot(basis, segmentation_probs_cur[j], color=list(colors[j][:-1]) + [0.5], linestyle=':', label='L%d source' %(j+1))
                    ax_s.plot(basis, segs_smoothed, color=colors[j], label='L%d smoothed' %(j+1))
                # _, _, w, _ = ax_seg_hard.get_window_extent().bounds
                # linewidth = w / len(segs_smoothed)
                y_start = float(j) / len(segmentations_cur) * n_feats
                y_end = float(j + 1) / len(segmentations_cur) * n_feats
                axes['input'].vlines(timestamps, y_start, y_end, color='k', linewidth=1)

            if has_smoothing:
                ax_s.set_xlim(0., basis[-1])
                ax_s.legend(fancybox=True, framealpha=0.75, frameon=True, facecolor='white', edgecolor='gray')
                ax_s.set_xlabel('Time')
                ax_s.set_title('Smoothed Segmentations')

        if has_states:
            for j in range(n_states):
                ax = axes['states_%d' % j]
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
                    cmap=get_cmap(j + 1),
                    ax=ax
                )
                ax.set_title('Hidden States (%s)' %(j+1))

        for k in plot_keys:
            for j, t in enumerate([targs, preds, attn, attn_keys, positional_encodings]):
                if t is not None and k in t:
                    plot_name = [
                        'Targets ',
                        'Predictions ',
                        'Top-Down Attention ',
                        'Top-Down Attention Keys ',
                        'Positional Encodings ',
                    ][j]
                    plot_name += k

                    l = int(re.match('L([0-9]+)', k).group(1)) - 1

                    ax = axes[plot_name]

                    arr = t[k][i]
                    if drop_zeros:
                        arr_select = np.where(np.any(arr[:,:-1] != 0., axis=1))[0]
                        arr = arr[arr_select]
                    arr = np.swapaxes(arr, -2, -1)

                    if arr.shape[-1] > 0:
                        librosa.display.specshow(
                            arr,
                            sr=sr,
                            hop_length=hop_length,
                            fmax=8000,
                            x_axis='time',
                            cmap=get_cmap(l),
                            ax=ax
                        )
                    ax.set_title(plot_name)

        if titles[i] is not None:
            title = titles[i]
            if label_map is not None:
                title = label_map.get(title, title)
            fig.suptitle(title, fontsize=20, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            fig.savefig(directory + '/' + prefix + '%d_featureplot' % i + suffix)
        except Exception:
            stderr('IO error when saving plot. Skipping plotting...\n')

        for ax in axes.values():
            ax.clear()

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
        stderr('IO error when saving plot. Skipping plotting...\n')

    plt.close(fig)


def plot_label_heatmap(seg_table, class_column_name='IPA', title=None, cmap='Blues', dir='./', prefix='', suffix='.png'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    labels = seg_table[class_column_name]
    preds = seg_table.label

    df = pd.crosstab(labels, preds)

    df = pd.DataFrame(np.array(df) / np.array(df).sum(axis=0, keepdims=True), index=df.index, columns=df.columns)
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
        stderr('IO error when saving plot. Skipping plotting...\n')

    plt.close(fig)


def plot_binary_unit_heatmap(seg_table, class_column_name='IPA', cmap='Blues', dir='./', prefix='', suffix='.png'):
    embedding_cols = [x for x in seg_table.columns if is_embedding_dimension.match(x)]
    df = seg_table[[class_column_name] + embedding_cols]
    df[class_column_name] += '   '
    # df = df[~df[class_column_name].isin(['SPN', 'SIL'])]
    df = df.groupby(class_column_name, as_index=True).mean()
    df.index.name = None

    width = 1 + .3 * len(df.columns)
    height = .27 * len(df.index)

    # Correlation-based clustering is undefined if any vectors happen to have no variance.
    # Try correlation first, then fall back to default distance metric if it fails.
    try:
        cm = sns.clustermap(data=df, metric='correlation', cmap=cmap, figsize=(width,height))
    except Exception:
        cm = sns.clustermap(data=df, cmap=cmap, figsize=(width,height))
    cm.cax.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)

    try:
        cm.savefig(dir + '/' + prefix + 'binary_unit_heatmap' + suffix)
    except Exception:
        stderr('IO error when saving plot. Skipping plotting...\n')

    plt.close('all')


def plot_class_similarity(
        similarity_matrix,
        title=None,
        cmap='Blues',
        dir='./',
        prefix='',
        suffix='.png'
):
    width = .3 * len(similarity_matrix.columns)
    height = .3 * len(similarity_matrix.index)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.set_size_inches(width, height)

    # sparse_sim = sparse.csc_matrix(similarity_matrix.values)
    # idx = sparse.csgraph.reverse_cuthill_mckee(sparse_sim, symmetric_mode=True)

    pdist = similarity_matrix.values
    dmin = pdist.min()
    dmax = pdist.max()
    drange = dmax - dmin
    pdist = (pdist - dmin) / drange
    linkage = spc.linkage(pdist, method='single')
    idx = spc.fcluster(linkage, 0.5 * pdist.max())

    classes = list(similarity_matrix.columns)
    classes = [classes[i] for i in list((np.argsort(idx)))]
    similarity_matrix = similarity_matrix.reindex(classes, axis=0)
    similarity_matrix = similarity_matrix.reindex(classes, axis=1)

    ax.pcolormesh(similarity_matrix, cmap=cmap, vmin=0., vmax=1.)
    ax.set_xticks(np.arange(0.5, len(similarity_matrix.columns), 1), minor=False)
    ax.set_xticklabels(similarity_matrix.columns)
    ax.set_yticks(np.arange(0.5, len(similarity_matrix.index), 1), minor=False)
    ax.set_yticklabels(similarity_matrix.index)

    if title is not None:
        fig.suptitle(title)
    # fig.tight_layout()

    try:
        fig.savefig(dir + '/' + prefix + 'confusion_matrix' + suffix)
    except:
        stderr('IO error when saving plot. Skipping plotting...\n')

    plt.close(fig)


def plot_projections(
        df,
        cmap='hls',
        label_map=None,
        feature_table=None,
        feature_names=None,
        directory='./',
        prefix='',
        suffix='.png'
):
    df = df.copy()

    df['CV'] = df.phn_label.map(lambda x: 'V' if (not x[0] in ['el', 'en'] and x[0] in ['a', 'e', 'i', 'o', 'u']) else 'C')

    if label_map is not None:
        df['IPA'] = df.phn_label.map(label_map)

    if feature_table is not None:
        if not 'gold_label' in feature_table.columns:
            feature_table['phn_label'] = feature_table.label
        df = df.merge(feature_table, on=['phn_label'])

    colors = ['CV']
    if 'IPA' in df.columns:
        colors.append('IPA')
    else:
        colors.append('phn_label')

    if feature_names is not None:
        new_colors = []
        for c in feature_names:
            if c in df:
                df[c] = df[c].map({0: '-', 1: '+'})
                new_colors.append(c)
        colors += new_colors

    df['Projection 1'] = df.projection1
    df['Projection 2'] = df.projection2

    for c in colors:
        sns.set_style('white')

        with sns.plotting_context(rc={'legend.fontsize': 'small', 'lines.markersize': 2}):
            g = sns.relplot(
                x='Projection 1',
                y='Projection 2',
                kind='scatter',
                hue=c,
                data=df.sort_values(c),
                palette=cmap,
                legend='full',
                size=0.5,
                alpha=0.5
            )
        try:
            g.savefig(directory + '/' + prefix + 'projections_%s' % (c + suffix))
        except Exception:
            stderr('IO error when saving plot. Skipping plotting...\n')

        plt.close('all')


