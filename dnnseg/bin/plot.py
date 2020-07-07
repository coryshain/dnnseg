import os
import re
import numpy as np
import pandas as pd
import argparse

from dnnseg.config import Config
from dnnseg.plot import run_classification_plots
from dnnseg.util import stderr


is_embedding_dimension = re.compile('d([0-9]+)')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate diagnostic plots from DNNSeg output tables. Primarily useful for phone-level acoustic models, since
    the lexical output space is often too large for plotting to be of practical use.
    ''')
    argparser.add_argument('config', nargs='+', help='Path(s) to config file(s) defining DNNSeg model(s).')
    argparser.add_argument('-s', '--seg_types', default=['phn'], nargs='+', help='Space-delimited list of segment types. Available types: "phn", "wrd".')
    argparser.add_argument('-p', '--plot_types', default=None, nargs='+', help='Space-delimited list of plot types. Available types: "unit_heatmap", "label_heatmap", "confusion_matrix", "projection". If unspecified, all available types will be generated.')
    argparser.add_argument('-c', '--class_limit', default='256', type=str, help='Largest number of allowed classes in plots by class. Plots that would exceed this limit will be skipped. Accepts an `int` or "inf", for no limit.')
    argparser.add_argument('-f', '--feature_names', default=['all'], nargs='+', help='Names of phonological features to use to projection plots.')
    argparser.add_argument('-m', '--projection_method', default='tsne', help='Embedding method to use for projections. One of ["lle", "mds", "pca", "spectral_embedding", "tsne"].')
    argparser.add_argument('-l', '--layers', default=None, nargs='+', help='IDs of layers to plot (0, ..., L). If unspecified, plots all available layers.')
    args = argparser.parse_args()

    if args.class_limit.lower() == 'inf':
        class_limit = np.inf
    else:
        class_limit = int(args.class_limit)

    for config_path in args.config:
        p = Config(config_path)
        stderr('Plotting model defined at path %s...\n' % p['outdir'])

        if p['label_map_file'] is not None and os.path.exists(p['label_map_file']):
            label_map = pd.read_csv(p['label_map_file'])
            label_map = dict(zip(label_map.source, label_map.target))
        else:
            label_map = None

        if p['feature_map_file'] is not None and os.path.exists(p['feature_map_file']):
            feature_table = pd.read_csv(p['feature_map_file'])
            new_cols = []
            for i in range(len(feature_table.columns)):
                new_cols.append(feature_table.columns[i].replace(' ', '_').replace('+', '-'))
            feature_table.columns = new_cols
            if not 'label' in feature_table.columns and 'symbol' in feature_table.columns:
                feature_table['label'] = feature_table.symbol
        else:
            feature_table = None

        if args.feature_names.lower() == 'all' and p['feature_map_file'].startswith('eng'):
            feature_names = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant',
                             'nasal', 'voice', 'spread_glottis', 'labial', 'round', 'labiodental', 'coronal',
                             'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back',
                             'tense', 'stress', 'diphthong']
        elif args.feature_names.lower() == 'all' and p['feature_map_file'].startswith('xit'):
            feature_names = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant',
                             'trill', 'nasal', 'voice', 'spread_glottis', 'constricted_glottis', 'labial', 'round',
                             'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal',
                             'high', 'low', 'front', 'back', 'tense', 'implosive']
        else:
            feature_names = args.feature_names

        for path in os.listdir(p['outdir'] + '/tables/'):
            if path.startswith('embeddings') and path.endswith('.csv'):
                layer_match = args.layers is None
                if not layer_match:
                    for l in args.layers:
                        if '_l%s' % l in path:
                            layer_match = True
                            break

                if layer_match:
                    stderr('  Plotting segment table: %s...\n' % path)
                    segs = pd.read_csv(os.path.join(p['outdir'] + '/tables/', path), sep=' ')

                    for s in args.seg_types:
                        run_classification_plots(
                            segs,
                            s,
                            plot_types=args.plot_types,
                            class_limit=args.class_limit,
                            projection_method=args.projection_method,
                            label_map=label_map,
                            feature_table=feature_table,
                            feature_names=feature_names,
                            outdir=os.path.join(p['outdir'] + '/tables/'),
                            prefix=path[:-4]
                        )