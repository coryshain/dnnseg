import sys
import math
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree  import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import f1_score
import pydot
import argparse

from dnnseg.data import get_random_permutation

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('data', help='Path to data containing predicted and gold classification labels.')
    argparser.add_argument('gold_cols', nargs='+', help='Names of column in data set to use as regression target.')
    argparser.add_argument('-n', '--n_estimators', type=int, default=100, help='Number of estimators (trees) in random forest.')
    argparser.add_argument('-f', '--n_folds', type=int, default=5, help='Number of folds in cross-validation (>= 2).')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Report progress to standard error.')
    argparser.add_argument('-o', '--outdir', default='./', help='Output directory.')
    args = argparser.parse_args()

    is_embedding_dimension = re.compile('d([0-9]+)')

    df = pd.read_csv(args.data)
    latent_dim_names = [c for c in df.columns if is_embedding_dimension.match(c)]
    X = df[latent_dim_names] > 0.5

    scores = {}

    for gold_col in args.gold_cols:
        perm, perm_inv = get_random_permutation(len(X))
        fold_size = math.ceil(float(len(X)) / args.n_folds)
        y = df[gold_col]

        score = 0
        predictions = []
        gold = []

        for i in range(0, len(X), fold_size):
            classifier = RandomForestClassifier(n_estimators=args.n_estimators, min_impurity_decrease=0.001)

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
        scores[gold_col] = f1_score(gold, predictions)

        if args.verbose:
            sys.stderr.write('Cross-validation F1 for variable "%s": %.4f\n' % (gold_col, scores[gold_col]))

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
        for c in sorted(list(scores.keys())):
            f.write('%s %s\n' % (c, scores[c]))
