import sys
import os
import math
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree  import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pydot
import argparse

from dnnseg.config import Config
from dnnseg.data import get_random_permutation
from dnnseg.util import stderr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', nargs='+', help='Path(s) to config file(s) defining DNNSeg model(s).')
    argparser.add_argument('-c', '--classes', nargs='+', default=['features'], help='Names of column in data set to use as regression target.')
    argparser.add_argument('-t', '--classifier_type', default='logreg', help='Type of classifier to use. One of ``["logreg", "random_forest"]``.')
    argparser.add_argument('-d', '--direction', type=str, default='pred2gold', help='Direction of classification. One of ["gold2pred", "pred2gold"].')
    argparser.add_argument('-r', '--regularization_scale', type=float, default=1., help='Level of regularization to use in MLR classification.')
    argparser.add_argument('-M', '--max_depth', type=float, default=None, help='Maximum permissible tree depth.')
    argparser.add_argument('-m', '--min_impurity_decrease', type=float, default=0., help='Minimum impurity decrease necessary to make a split.')
    argparser.add_argument('-n', '--n_estimators', type=int, default=100, help='Number of estimators (trees) in random forest.')
    argparser.add_argument('-f', '--n_folds', type=int, default=5, help='Number of folds in cross-validation (>= 2).')
    argparser.add_argument('-l', '--layers', default=None, nargs='+', help='IDs of layers to plot (0, ..., L). If unspecified, plots all available layers.')
    argparser.add_argument('-b', '--compare_to_baseline', action='store_true', help='Whether to compute scores for a matched random baseline.')
    argparser.add_argument('-i', '--images', action='store_true', help='Dump representative images of decision trees.')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Report progress to standard error.')
    args = argparser.parse_args()

    is_embedding_dimension = re.compile('d([0-9]+)')

    for config_path in args.config:
        p = Config(config_path)

        if p['label_map_file'] is not None and os.path.exists(p['label_map_file']):
            label_map = pd.read_csv(p['label_map_file'])
            label_map = dict(zip(label_map.source, label_map.target))
        else:
            label_map = None

        if p['feature_map_file'] is not None and os.path.exists(p['feature_map_file']):
            feature_table = pd.read_csv(p['feature_map_file'])
            new_cols = []
            for j in range(len(feature_table.columns)):
                new_cols.append(feature_table.columns[j].replace(' ', '_').replace('+', '-'))
            feature_table.columns = new_cols
            if not 'label' in feature_table.columns and 'symbol' in feature_table.columns:
                feature_table['label'] = feature_table.symbol
        else:
            feature_table = None

        if args.classes == ['features']:
            if p['feature_map_file'].startswith('english'):
                target_col_names = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant',
                                 'nasal', 'voice', 'spread_glottis', 'labial', 'round', 'labiodental', 'coronal',
                                 'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back',
                                 'tense', 'stress', 'diphthong']
            elif p['feature_map_file'].startswith('xitsonga'):
                target_col_names = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant',
                                 'trill', 'nasal', 'voice', 'spread_glottis', 'constricted_glottis', 'labial', 'round',
                                 'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal',
                                 'high', 'low', 'front', 'back', 'tense', 'implosive']
            else:
                target_col_names = args.classes
        else:
            target_col_names = args.classes

        analysis_dir = os.path.join(p['outdir'], 'feature_analysis')
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)

        for path in os.listdir(p['outdir']):
            if path.startswith('embeddings') and path.endswith('.csv'):
                layer_match = args.layers is None
                if not layer_match:
                    for l in args.layers:
                        if '_l%s' % l in path:
                            layer_match = True
                            break

                if layer_match:
                    if args.verbose:
                        sys.stderr.write('Evaluating segment table %s...\n' % path)
                        sys.stderr.flush()
                    df = pd.read_csv(os.path.join(p['outdir'], path), sep=' ')
                    input_col_names = [c for c in df.columns if is_embedding_dimension.match(c)]
                    target_col_names_cur = []
                    df_cols = set(df.columns)
                    for target_col in target_col_names:
                        if target_col in df_cols:
                            # if len(df[target_col].unique()) > 2:
                            #     dummies = pd.get_dummies(df[[target_col]])
                            #     df = pd.concat([df, dummies], axis=1)
                            #     target_col_names_cur += list(dummies.columns)
                            # else:
                            #     target_col_names_cur.append(target_col)
                            target_col_names_cur.append(target_col)

                    precision = {}
                    recall = {}
                    f1 = {}
                    accuracy = {}

                    if args.direction.lower() == 'pred2gold':
                        X = df[input_col_names] > 0.5
                        if args.compare_to_baseline:
                            base_rates = X.values.sum(axis=0, keepdims=True)
                            X_baseline = np.random.random(X.shape) > base_rates
                            inputs = [X, X_baseline]
                        else:
                            inputs = [X]

                        for i, X_cur in enumerate(inputs):
                            if args.verbose:
                                if i == 0:
                                    sys.stderr.write('Evaluating model...\n')
                                else:
                                    sys.stderr.write('Evaluating baseline...\n')
                                sys.stderr.flush()
                            if len(target_col_names_cur):
                                for target_col in target_col_names_cur:
                                    perm, perm_inv = get_random_permutation(len(X))
                                    fold_size = math.ceil(float(len(X)) / args.n_folds)
                                    y = df[target_col]
                                    if len(y.unique()) > 2:
                                        avg_method = 'macro'
                                    else:
                                        avg_method = 'binary'

                                    score = 0
                                    predictions = []
                                    gold = []

                                    for j in range(0, len(X_cur), fold_size):
                                        if args.classifier_type.lower() == 'random_forest':
                                            classifier = RandomForestClassifier(
                                                n_estimators=args.n_estimators,
                                                criterion='entropy',
                                                class_weight='balanced',
                                                max_depth=args.max_depth,
                                                min_impurity_decrease=args.min_impurity_decrease
                                            )
                                        elif args.classifier_type.lower() in ['mlr', 'logreg', 'logistic_regression']:
                                            classifier = LogisticRegression(
                                                class_weight='balanced',
                                                C = args.regularization_scale,
                                                solver='lbfgs',
                                                multi_class='auto'
                                            )

                                        train_select = np.zeros(len(X_cur)).astype('bool')
                                        train_select[j:j + fold_size] = True
                                        train_select = train_select[perm_inv]

                                        cv_select = np.ones(len(X_cur)).astype('bool')
                                        cv_select[j:j + fold_size] = False
                                        cv_select = cv_select[perm_inv]

                                        X_train = X_cur[train_select]
                                        y_train = y[train_select]
                                        X_cv = X[cv_select]
                                        y_cv = y[cv_select]

                                        classifier.fit(X_train, y_train)
                                        predictions.append(classifier.predict(X_cv))
                                        gold.append(y_cv)

                                    predictions = np.concatenate(predictions, axis=0)
                                    gold = np.concatenate(gold, axis=0)
                                    precision[target_col] = precision_score(gold, predictions, average=avg_method)
                                    recall[target_col] = recall_score(gold, predictions, average=avg_method)
                                    f1[target_col] = f1_score(gold, predictions, average=avg_method)
                                    accuracy[target_col] = accuracy_score(gold, predictions)

                                    if args.verbose:
                                        stderr('Cross-validation F1 for variable "%s": %.4f\n' % (target_col, f1[target_col]))

                                    if args.images:
                                        tree_ix = np.random.randint(args.n_estimators)

                                        graph = export_graphviz(
                                            classifier[tree_ix],
                                            feature_names=input_col_names,
                                            class_names=['-%s' % target_col, '+%s' % target_col],
                                            rounded=True,
                                            proportion=False,
                                            precision=2,
                                            filled=True
                                        )

                                        (graph,) = pydot.graph_from_dot_data(graph)

                                        if i == 0:
                                            img_str = '/%s_decision_tree_%s.png'
                                        else:
                                            img_str = '/%s_decision_tree_%s_rand.png'

                                        outfile = analysis_dir + img_str % (path[:-4], target_col)
                                        graph.write_png(outfile)

                                macro_avg = {
                                    'precision': sum(precision[x] for x in precision) / sum(1 for _ in precision),
                                    'recall': sum(recall[x] for x in recall) / sum(1 for _ in recall),
                                    'f1': sum(f1[x] for x in f1) / sum(1 for _ in f1),
                                    'accuracy': sum(accuracy[x] for x in accuracy) / sum(1 for _ in accuracy),
                                }

                                if args.verbose:
                                    stderr('Macro averages:\n')
                                    stderr('  P:   %.4f\n' % macro_avg['precision'])
                                    stderr('  R:   %.4f\n' % macro_avg['recall'])
                                    stderr('  F1:  %.4f\n' % macro_avg['f1'])
                                    stderr('  ACC: %.4f\n' % macro_avg['accuracy'])

                                if i == 0:
                                    path_str = '/%s_decision_tree_scores.txt'
                                else:
                                    path_str = '/%s_decision_tree_scores_rand.txt'

                                outfile = analysis_dir + path_str % path[:-4]
                                with open(outfile, 'w') as f:
                                    f.write('feature precision recall f1 accuracy\n')
                                    for c in sorted(list(f1.keys())):
                                        f.write('%s %s %s %s %s\n' % (c, precision[c], recall[c], f1[c], accuracy[c]))
                                    f.write('MACRO %s %s %s %s\n' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1'], macro_avg['accuracy']))

                    elif args.direction.lower() == 'gold2pred':
                        X = df[target_col_names_cur] > 0.5
                        y = df[input_col_names]
                        if args.compare_to_baseline:
                            base_rates = y.values.sum(axis=0, keepdims=True)
                            y_baseline = np.random.random(y.shape) > base_rates
                            targets = [y, y_baseline]
                        else:
                            targets = [y]

                        if len(input_col_names) > 0:
                            for target in targets:
                                for latent_dim in input_col_names:
                                    perm, perm_inv = get_random_permutation(len(X))
                                    fold_size = math.ceil(float(len(X)) / args.n_folds)
                                    y_cur = target[latent_dim]
                                    score = 0
                                    predictions = []
                                    gold = []

                                    for j in range(0, len(X), fold_size):
                                        if args.classifier_type.lower() == 'random_forest':
                                            classifier = RandomForestClassifier(
                                                n_estimators=args.n_estimators,
                                                criterion='entropy',
                                                class_weight='balanced',
                                                max_depth=args.max_depth,
                                                min_impurity_decrease=args.min_impurity_decrease
                                            )
                                        elif args.classifier_type.lower() in ['mlr', 'logreg', 'logistic_regression']:
                                            classifier = LogisticRegression(
                                                class_weight='balanced',
                                                C = args.regularization_scale,
                                                solver='lbfgs',
                                                multi_class='auto'
                                            )

                                        train_select = np.zeros(len(X)).astype('bool')
                                        train_select[j:j + fold_size] = True
                                        train_select = train_select[perm_inv]

                                        cv_select = np.ones(len(X)).astype('bool')
                                        cv_select[j:j + fold_size] = False
                                        cv_select = cv_select[perm_inv]

                                        X_train = X[train_select]
                                        y_train = y_cur[train_select]
                                        X_cv = X[cv_select]
                                        y_cv = y_cur[cv_select]

                                        classifier.fit(X_train, y_train)
                                        predictions.append(classifier.predict(X_cv))
                                        gold.append(y_cv)

                                    predictions = np.concatenate(predictions, axis=0)
                                    gold = np.concatenate(gold, axis=0)
                                    precision[latent_dim] = precision_score(gold, predictions)
                                    recall[latent_dim] = recall_score(gold, predictions)
                                    f1[latent_dim] = f1_score(gold, predictions)
                                    accuracy[latent_dim] = accuracy_score(gold, predictions)

                                    if args.verbose:
                                        stderr('Cross-validation F1 for latent dimension "%s": %.4f\n' % (latent_dim, f1[latent_dim]))

                                    if args.images:
                                        tree_ix = np.random.randint(args.n_estimators)

                                        graph = export_graphviz(
                                            classifier[tree_ix],
                                            feature_names=target_col_names,
                                            class_names=['-%s' % latent_dim, '+%s' % latent_dim],
                                            rounded=True,
                                            proportion=False,
                                            precision=2,
                                            filled=True
                                        )

                                        (graph,) = pydot.graph_from_dot_data(graph)

                                        outfile = analysis_dir + analysis_dir + '/%s_decision_tree_%s.png' % (path[:-4], latent_dim)
                                        graph.write_png(outfile)

                                macro_avg = {
                                    'precision': sum(precision[x] for x in precision) / sum(1 for _ in precision),
                                    'recall': sum(recall[x] for x in recall) / sum(1 for _ in recall),
                                    'f1': sum(f1[x] for x in f1) / sum(1 for _ in f1),
                                    'accuracy': sum(accuracy[x] for x in accuracy) / sum(1 for _ in accuracy),
                                }

                                if args.verbose:
                                    stderr('Macro averages:\n')
                                    stderr('  P:   %.4f\n' % macro_avg['precision'])
                                    stderr('  R:   %.4f\n' % macro_avg['recall'])
                                    stderr('  F1:  %.4f\n' % macro_avg['f1'])
                                    stderr('  ACC: %.4f\n' % macro_avg['accuracy'])

                                outfile = analysis_dir + '/%s_decision_tree_scores.txt' % path[:-4]
                                with open(outfile, 'w') as f:
                                    f.write('feature precision recall f1 accuracy\n')
                                    for c in sorted(list(f1.keys())):
                                        f.write('%s %s %s %s %s\n' % (c, precision[c], recall[c], f1[c], accuracy[c]))
                                    f.write('MACRO %s %s %s %s\n' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1'], macro_avg['accuracy']))

                    else:
                        raise ValueError('Direction parameter %s not recognized.' % args.direction)
