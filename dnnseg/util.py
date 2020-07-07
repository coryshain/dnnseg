import sys
import re
import os
import pickle


def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def sn(string):
    """
    Compute a Tensorboard-compatible version of a string.

    :param string: ``str``; input string
    :return: ``str``; transformed string
    """

    return re.sub('[^A-Za-z0-9_.\\-/]', '.', string)


def f_measure(tp, fp, fn, beta=1):
    p = r = f = 0
    if sum([tp, fp]) > 0:
        p = tp / (tp + fp)
    if sum([tp, fn]) > 0:
        r = tp / (tp + fn)
    if sum([p, r]) > 0:
        f = (1 + beta) * (p * r / (p + r))

    return p, r, f


def load_dnnseg(dir_path):
    """
    Convenience method for reconstructing a saved DNNSeg object. First loads in metadata from ``m.obj``, then uses
    that metadata to construct the computation graph. Then, if saved weights are found, these are loaded into the
    graph.

    :param dir_path: Path to root directory for DNNSeg model.
    :return: The loaded DNNSeg instance.
    """

    with open(dir_path + '/model/m.obj', 'rb') as f:
        m = pickle.load(f)
    m.build(outdir=dir_path)
    m.load(outdir=dir_path)
    return m


def pretty_print_seconds(s):
    s = int(s)
    h = s // 3600
    m = s % 3600 // 60
    s = s % 3600 % 60
    return '%02d:%02d:%02d' % (h, m, s)


# Thanks to SO user gbtimmon (https://stackoverflow.com/users/1158990/gbtimmon) for this wrapper to prevent modules from printing
def suppress_output(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull,"w") as devNull:
            original = sys.stdout
            sys.stdout = devNull
            out = func(*args, **kwargs)
            sys.stdout = original
            return out
    return wrapper


def get_alternating_mode(x, r):
    t, m = r.as_integer_ratio()
    if x % (t + m) < t:
        return 0 # first mode
    else:
        return 1 # second mode