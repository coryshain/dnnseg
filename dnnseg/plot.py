import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt

def plot_acoustic_features(inputs, targets, preds, sr=16000, hop_length=160, cmap='Blues', title=None, dir='./', prefix='', suffix='.png'):

    fig = plt.figure()

    fig.set_size_inches(10, 10)
    ax_input = fig.add_subplot(311)
    ax_targ = fig.add_subplot(312)
    ax_pred = fig.add_subplot(313)

    inputs = np.swapaxes(inputs, -2, -1)
    targets = np.swapaxes(targets, -2, -1)
    preds = np.swapaxes(preds, -2, -1)

    while len(inputs.shape) < 3:
        inputs = np.expand_dims(inputs, 0)

    while len(targets.shape) < 3:
        targets = np.expand_dims(targets, 0)

    while len(preds.shape) < 3:
        preds = np.expand_dims(preds, 0)

    for i in range(inputs.shape[0]):

        librosa.display.specshow(
            inputs[i],
            sr=sr,
            hop_length=hop_length,
            fmax=8000,
            x_axis='time',
            cmap=cmap,
            ax=ax_input
        )
        ax_input.set_title('Inputs')

        librosa.display.specshow(
            targets[i],
            sr=sr,
            hop_length=hop_length,
            fmax=8000,
            x_axis='time',
            cmap=cmap,
            ax=ax_targ
        )
        ax_targ.set_title('Targets')

        librosa.display.specshow(
            preds[i],
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
        fig.savefig(dir + '/' + prefix + 'featureplot_%d'%i + suffix)

    plt.close(fig)


def plot_label_histogram(labels, title=None, bins='auto', dir='./', prefix='', suffix='.png'):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(labels, bins=bins)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(dir + '/' + prefix + 'label_histogram' + suffix)

    plt.close(fig)


