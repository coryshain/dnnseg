import sys
import os
import math
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree  import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pydot
import argparse

from dnnseg.config import Config
from dnnseg.probe import probe


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', nargs='+', help='Path(s) to config file(s) defining DNNSeg model(s).')
    argparser.add_argument('-c', '--classes', nargs='+', default=['features'], help='Names of column(s) in data set to use as regression target(s).')
    argparser.add_argument('-t', '--classifier_type', default='mlp', help='Type of classifier to use. One of ``["logreg", "mlp", "random_forest"]``.')
    argparser.add_argument('-r', '--regularization_scale', type=float, default=0.001, help='Level of regularization to use in MLR classification.')
    argparser.add_argument('-M', '--max_depth', type=float, default=None, help='Maximum permissible tree depth.')
    argparser.add_argument('-m', '--min_impurity_decrease', type=float, default=0., help='Minimum impurity decrease necessary to make a split.')
    argparser.add_argument('-n', '--n_estimators', type=int, default=100, help='Number of estimators (trees) in random forest.')
    argparser.add_argument('-f', '--n_folds', type=int, default=2, help='Number of folds in cross-validation (>= 2).')
    argparser.add_argument('-l', '--layers', default=None, nargs='+', help='IDs of layers to plot (0, ..., L). If unspecified, plots all available layers.')
    argparser.add_argument('-u', '--units', default=[100], nargs='+', help='Space-delimited list of integers representing number of hidden units to use in MLP classifier.')
    argparser.add_argument('-b', '--compare_to_baseline', action='store_true', help='Whether to compute scores for a matched random baseline.')
    argparser.add_argument('-i', '--images', action='store_true', help='Dump representative images of decision trees.')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Report progress to standard error.')
    args = argparser.parse_args()

    is_embedding_dimension = re.compile('d([0-9]+)')

    units = [int(x) for x in args.units]

    class_types = args.classes

    for config_path in args.config:
        p = Config(config_path)

        analysis_dir = os.path.join(p['outdir'], 'feature_analysis')

        for path in os.listdir(p['outdir'] + '/tables/'):
            if path.startswith('embeddings') and path.endswith('.csv'):
                layer_match = args.layers is None
                if not layer_match:
                    for l in args.layers:
                        if '_l%s' % l in path:
                            layer_match = True
                            break

                if layer_match:
                    if args.verbose:
                        sys.stderr.write('*' * 50 + '\n')
                        sys.stderr.write('Analyzing segment table %s...\n' % path)
                        sys.stderr.flush()

                    df = pd.read_csv(os.path.join(p['outdir'] + '/tables/', path), sep=' ')

                    if p['feature_map_file'] is not None and p['feature_map_file'].lower().startswith('eng'):
                        lang='english'
                    elif p['feature_map_file'] is not None and p['feature_map_file'].lower().startswith('xit'):
                        lang='xitsonga'
                    else:
                        lang=None

                    probe(
                        df,
                        class_types,
                        lang=lang,
                        classifier_type=args.classifier_type,
                        regularization_scale=args.regularization_scale,
                        max_depth=args.max_depth,
                        min_impurity_decrease=args.min_impurity_decrease,
                        n_estimators=args.n_estimators,
                        n_folds=args.n_folds,
                        units=args.units,
                        compare_to_baseline=args.compare_to_baseline,
                        dump_images=args.images,
                        verbose=args.verbose,
                        outdir=analysis_dir,
                        name=path[:-4]
                    )