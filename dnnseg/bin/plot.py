import os
import re
import numpy as np
import pandas as pd
import argparse

from dnnseg.config import Config
from dnnseg.data import project_matching_segments, compute_class_similarity
from dnnseg.plot import plot_binary_unit_heatmap, plot_label_heatmap, plot_class_similarity, plot_projections
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
    argparser.add_argument('-f', '--feature_names', default=['english'], nargs='+', help='Names of phonological features to use to projection plots.')
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

        if args.feature_names == ['english']:
            feature_names = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant',
                             'nasal', 'voice', 'spread_glottis', 'labial', 'round', 'labiodental', 'coronal',
                             'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back',
                             'tense', 'stress', 'diphthong']
        elif args.feature_names == ['xitsonga']:
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
                        stderr('    Plotting segment type: %s...\n' % s)
                        gold_col = s + '_label'
                        if gold_col in segs.columns and 'label' in segs.columns:
                            embedding_cols = [x for x in segs.columns if is_embedding_dimension.match(x)]
                            unique_classes = segs[gold_col].unique()
                            unique_labels = segs['label'].unique()

                            outdir = p['outdir'] + '/plots'
                            prefix = path[:-4] + '_'
                            if not os.path.exists(outdir):
                                os.makedirs(outdir)

                            if gold_col == 'phn_label' and 'IPA' in segs.columns:
                                gold_col = 'IPA'

                            if args.plot_types is None or 'unit_heatmap'.lower() in args.plot_types:
                                stderr('      Plotting unit heatmap...\n')
                                if len(embedding_cols) < class_limit and len(unique_classes) < class_limit:
                                    plot_binary_unit_heatmap(
                                        segs,
                                        class_column_name=gold_col,
                                        directory=outdir + '/classification/',
                                        prefix=prefix,
                                        suffix='.png')

                            if args.plot_types is None or 'label_heatmap'.lower() in args.plot_types:
                                if len(embedding_cols) < class_limit and len(unique_labels) < class_limit:
                                    stderr('      Plotting label heatmap...\n')
                                    plot_label_heatmap(
                                        segs,
                                        class_column_name=gold_col,
                                        directory=outdir + '/classification/',
                                        prefix=prefix,
                                        suffix='.png')

                            if args.plot_types is None or 'confusion_matrix'.lower() in args.plot_types:
                                if len(unique_classes) < class_limit:
                                    stderr('      Plotting confusion matrix...\n')
                                    sim = compute_class_similarity(segs, class_column_name=gold_col)
                                    plot_class_similarity(
                                        sim,
                                        directory=outdir + '/classification/',
                                        prefix=prefix,
                                    )

                            if (args.plot_types is None or 'projection'.lower() in args.plot_types):
                                stderr('      Plotting segment projections...\n')

                                projections = project_matching_segments(segs, method=args.projection_method)

                                plot_projections(
                                    projections,
                                    label_map=label_map,
                                    feature_table=feature_table,
                                    feature_names=feature_names,
                                    directory=outdir + '/projections/',
                                    prefix=prefix + '_%s_' % args.projection_method,
                                    suffix='.png'
                                )
