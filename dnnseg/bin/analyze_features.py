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

from dnnseg.data import get_random_permutation

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('data', help='Path to data containing predicted and gold classification labels.')
    argparser.add_argument('gold_cols', nargs='+', help='Names of column in data set to use as regression target.')
    argparser.add_argument('-l', '--language', default='English', help='Language to extract features for.')
    argparser.add_argument('-d', '--direction', type=str, default='pred2gold', help='Direction of classification. One of ["gold2pred", "pred2gold"].')
    argparser.add_argument('-M', '--max_depth', type=float, default=None, help='Maximum permissible tree depth.')
    argparser.add_argument('-m', '--min_impurity_decrease', type=float, default=0., help='Minimum impurity decrease necessary to make a split.')
    argparser.add_argument('-n', '--n_estimators', type=int, default=100, help='Number of estimators (trees) in random forest.')
    argparser.add_argument('-f', '--n_folds', type=int, default=5, help='Number of folds in cross-validation (>= 2).')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Report progress to standard error.')
    argparser.add_argument('-o', '--outdir', default='./', help='Output directory.')
    args = argparser.parse_args()

    is_embedding_dimension = re.compile('d([0-9]+)')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    df = pd.read_csv(args.data, sep=' ')
    if args.language.lower() == 'english':
        gold_feats = pd.read_csv('english_sampa_to_feats.csv')
    elif args.language.lower() == 'xitsonga':
        gold_feats = pd.read_csv('xitsonga_sampa_to_feats.csv')
    else:
        raise ValueError('Unsupported language "%s"' % args.language)

    gold_feats['gold_label'] = gold_feats.symbol

    df = df.merge(gold_feats, on='gold_label')

    latent_dim_names = [c for c in df.columns if is_embedding_dimension.match(c)]

    precision = {}
    recall = {}
    f1 = {}
    accuracy = {}

    if args.gold_cols == ['english']:
        gold_cols = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant', 'nasal', 'voice', 'spread_glottis', 'labial', 'round', 'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back', 'tense', 'stress', 'diphthong']
    elif args.gold_cols == ['xitsonga']:
        gold_cols = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant', 'trill', 'nasal', 'voice', 'spread_glottis', 'constricted_glottis', 'labial', 'round', 'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back', 'tense', 'implosive']
    else:
        gold_cols = args.gold_cols

    gold_cols = [c for c in gold_cols if c in df.columns]

    if args.direction.lower() == 'pred2gold':
        X = df[latent_dim_names] > 0.5

        for gold_col in gold_cols:
            perm, perm_inv = get_random_permutation(len(X))
            fold_size = math.ceil(float(len(X)) / args.n_folds)
            y = df[gold_col]

            score = 0
            predictions = []
            gold = []

            for i in range(0, len(X), fold_size):
                classifier = RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    criterion='entropy',
                    class_weight='balanced',
                    max_depth=args.max_depth,
                    min_impurity_decrease=args.min_impurity_decrease
                )

                train_select = np.zeros(len(X)).astype('bool')
                train_select[i:i+fold_size] = True
                train_select = train_select[perm_inv]

                cv_select = np.ones(len(X)).astype('bool')
                cv_select[i:i+fold_size] = False
                cv_select = cv_select[perm_inv]


                X_train = X[train_select]
                y_train = y[train_select]
                X_cv = X[cv_select]
                y_cv = y[cv_select]

                classifier.fit(X_train, y_train)
                predictions.append(classifier.predict(X_cv))
                gold.append(y_cv)

            predictions = np.concatenate(predictions, axis=0)
            gold = np.concatenate(gold, axis=0)
            precision[gold_col] = precision_score(gold, predictions)
            recall[gold_col] = recall_score(gold, predictions)
            f1[gold_col] = f1_score(gold, predictions)
            accuracy[gold_col] = accuracy_score(gold, predictions)

            if args.verbose:
                sys.stderr.write('Cross-validation F1 for variable "%s": %.4f\n' % (gold_col, f1[gold_col]))

            tree_ix = np.random.randint(args.n_estimators)

            graph = export_graphviz(
                classifier[tree_ix],
                feature_names=latent_dim_names,
                class_names=['-%s' % gold_col, '+%s' % gold_col],
                rounded=True,
                proportion=False,
                precision=2,
                filled=True
            )

            (graph,) = pydot.graph_from_dot_data(graph)

            outfile = args.outdir + '/decision_tree_%s.png' % gold_col
            graph.write_png(outfile)

        outfile = args.outdir + '/decision_tree_scores.txt'
        with open(outfile, 'w') as f:
            f.write('feature precision recall f1 accuracy\n')
            for c in sorted(list(f1.keys())):
                f.write('%s %s %s %s %s\n' % (c, precision[c], recall[c], f1[c], accuracy[c]))

    elif args.direction.lower() == 'gold2pred':
        X = df[gold_cols] > 0.5

        for latent_dim in latent_dim_names:
            perm, perm_inv = get_random_permutation(len(X))
            fold_size = math.ceil(float(len(X)) / args.n_folds)
            y = df[latent_dim]

            score = 0
            predictions = []
            gold = []

            for i in range(0, len(X), fold_size):
                classifier = RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    criterion='entropy',
                    class_weight='balanced',
                    max_depth=args.max_depth,
                    min_impurity_decrease=args.min_impurity_decrease
                )

                # classifier = LogisticRegression(
                #     class_weight='balanced',
                #     C = 0.1
                # )

                train_select = np.zeros(len(X)).astype('bool')
                train_select[i:i+fold_size] = True
                train_select = train_select[perm_inv]

                cv_select = np.ones(len(X)).astype('bool')
                cv_select[i:i+fold_size] = False
                cv_select = cv_select[perm_inv]


                X_train = X[train_select]
                y_train = y[train_select]
                X_cv = X[cv_select]
                y_cv = y[cv_select]

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
                sys.stderr.write('Cross-validation F1 for latent dimension "%s": %.4f\n' % (latent_dim, f1[latent_dim]))

            tree_ix = np.random.randint(args.n_estimators)

            graph = export_graphviz(
                classifier[tree_ix],
                feature_names=gold_cols,
                class_names=['-%s' % latent_dim, '+%s' % latent_dim],
                rounded=True,
                proportion=False,
                precision=2,
                filled=True
            )

            (graph,) = pydot.graph_from_dot_data(graph)

            outfile = args.outdir + '/decision_tree_%s.png' % latent_dim
            graph.write_png(outfile)

        outfile = args.outdir + '/decision_tree_scores.txt'
        with open(outfile, 'w') as f:
            f.write('feature precision recall f1 accuracy\n')
            for c in sorted(list(f1.keys())):
                f.write('%s %s %s %s %s\n' % (c, precision[c], recall[c], f1[c], accuracy[c]))

    else:
        raise ValueError('Direction parameter %s not recognized.' % args.direction)
