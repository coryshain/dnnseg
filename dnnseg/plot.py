import sys
import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt

def plot_acoustic_features(
        inputs,
        targets,
        preds,
        sr=16000,
        hop_length=160,
        cmap='Blues',
        title=None,
        dir='./',
        prefix='',
        suffix='.png'
):

    fig = plt.figure()

    fig.set_size_inches(10, 10)
    ax_input = fig.add_subplot(311)
    ax_targ = fig.add_subplot(312)
    ax_pred = fig.add_subplot(313)

    while len(inputs.shape) < 3:
        inputs = np.expand_dims(inputs, 0)

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

        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
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


