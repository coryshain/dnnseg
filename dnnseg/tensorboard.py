# Thanks to willwhitney (https://gist.github.com/willwhitney) for this script

from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
import os
import glob
import pandas as pd

tf.logging.set_verbosity(tf.logging.ERROR)


def load_tb(dirname):
    ea = event_accumulator.EventAccumulator(dirname, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    dframes = {}
    mnames = ea.Tags()['scalars']

    for n in mnames:
        dframes[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "step", '_'.join(n.split('/'))])
        dframes[n].drop("wall_time", axis=1, inplace=True)
        dframes[n].set_index("step", drop=True, inplace=True)
    return pd.concat([v for k, v in dframes.items()], axis=1)


def load_tbs(*root):
    dirs = get_tb_dirs(*root)
    out = {}
    for d in dirs:
        df = load_tb(d)
        out[d] = df

    return out


def get_tb_dirs(*paths):
    out = []
    for p in paths:
        dirs = []
        for r, d, f in os.walk(p):
            for x in d:
                dirs.append(os.path.join(r, x))
        out += [x for x in dirs if len([y for y in os.listdir(x) if 'tfevents' in y])]

    return out


def dump_tbs_to_csv(*root):
    dirs = get_tb_dirs(*root)
    for d in dirs:
        df = load_tb(d)
        df.to_csv(d + '/tb.csv', index=True, na_rep='NaN')
