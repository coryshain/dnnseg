import pickle

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

    :param dir_path: Path to directory containing the DTSR checkpoint files.
    :return: The loaded DTSR instance.
    """

    with open(dir_path + '/m.obj', 'rb') as f:
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
