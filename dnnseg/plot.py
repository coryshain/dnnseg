import sys
import numpy as np
import librosa
import librosa.display
import pandas as pd
from matplotlib import pyplot as plt

def plot_acoustic_features(
        inputs,
        targets,
        preds,
        target_means=None,
        sr=16000,
        hop_length=160,
        cmap='Blues',
        titles=None,
        dir='./',
        prefix='',
        suffix='.png'
):

    if titles is None:
        titles = [None] * inputs.shape[0]

    fig = plt.figure()

    if target_means is None:
        fig.set_size_inches(10, 10)
        ax_input = fig.add_subplot(311)
        ax_targ = fig.add_subplot(312)
        ax_pred = fig.add_subplot(313)
    else:
        fig.set_size_inches(10, 12)
        ax_input = fig.add_subplot(411)
        ax_targ_means = fig.add_subplot(412)
        ax_targ = fig.add_subplot(413)
        ax_pred = fig.add_subplot(414)

    while len(inputs.shape) < 3:
        inputs = np.expand_dims(inputs, 0)

    if target_means is not None:
        while len(target_means.shape) < 3:
            target_means = np.expand_dims(target_means, 0)

    while len(targets.shape) < 3:
        targets = np.expand_dims(targets, 0)

    while len(preds.shape) < 3:
        preds = np.expand_dims(preds, 0)

    for i in range(inputs.shape[0]):
        inputs_select = np.where(np.all(np.logical_not(np.isclose(inputs[i], 0.)), axis=1))[0]
        inputs_cur = np.swapaxes(inputs[i][inputs_select], -2, -1)

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

        if target_means is not None:
            target_means_select = np.where(np.all(np.logical_not(np.isclose(target_means[i], 0.)), axis=1))[0]
            target_means_cur = np.swapaxes(target_means[i][target_means_select], -2, -1)

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

        targets_select = np.where(np.all(np.logical_not(np.isclose(targets[i], 0.)), axis=1))[0]
        targets_cur = np.swapaxes(targets[i][targets_select], -2, -1)

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

        preds_select = np.where(np.all(np.logical_not(np.isclose(preds[i], 0.)), axis=1))[0]
        preds_cur = np.swapaxes(preds[i][preds_select], -2, -1)

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
            fig.suptitle(titles[i], fontsize=20, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            fig.savefig(dir + '/' + prefix + 'featureplot_%d'%i + suffix)
        except:
            sys.stderr.write('IO error when saving plot. Skipping plotting...\n')

    plt.close(fig)


def plot_label_histogram(labels, title=None, bins='auto', dir='./', prefix='', suffix='.png'):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(labels, bins=bins)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    try:
        fig.savefig(dir + '/' + prefix + 'label_histogram' + suffix)
    except:
        sys.stderr.write('IO error when saving plot. Skipping plotting...\n')

    plt.close(fig)

def plot_label_heatmap(labels, preds, title=None, cmap='Blues', dir='./', prefix='', suffix='.png'):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    df = pd.crosstab(labels, preds)
    df = pd.DataFrame(np.array(df) / np.array(df).sum(axis=0, keepdims=True), index=df.index, columns=df.columns)

    fig.set_size_inches(1 + .5 * len(df.columns), .25 * len(df.index))

    ax.pcolor(df, cmap=cmap)
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

def plot_binary_unit_heatmap(labels, label_probs, title=None, cmap='Blues', dir='./', prefix='', suffix='.png'):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    df = {'label': labels}
    for bit in range(label_probs.shape[1]):
        df['%d' %bit] = label_probs[:,bit]

    df = pd.DataFrame(df)
    df = df.groupby('label').mean()

    fig.set_size_inches(1 + .5 * len(df.columns), .25 * len(df.index))

    ax.pcolor(df, cmap=cmap, vmin=0., vmax=1.)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1), minor=False)
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(0.5, len(df.index), 1), minor=False)
    ax.set_yticklabels(df.index)

    if title is not None:
        fig.suptitle(title)
    # fig.tight_layout()

    try:
        fig.savefig(dir + '/' + prefix + 'binary_unit_heatmap' + suffix)
    except:
        sys.stderr.write('IO error when saving plot. Skipping plotting...\n')

    plt.close(fig)


