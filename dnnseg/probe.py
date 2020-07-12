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

from dnnseg.data import get_random_permutation, is_embedding_dimension
from dnnseg.util import stderr


def get_target_cols(
        class_types=None,
        lang=None,
):
    if class_types is None:
        class_types = []
    elif isinstance(class_types, str):
        class_types = class_types.split()
    elif not (isinstance(class_types, list) or isinstance(class_types, tuple)):
        class_types = [class_types]

    out = []

    for class_type in class_types:
        if class_type == 'features':
            if lang.lower().startswith('eng'):
                target_col_names = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant',
                                    'nasal', 'voice', 'spread_glottis', 'labial', 'round', 'labiodental', 'coronal',
                                    'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front',
                                    'back',
                                    'tense', 'stress', 'diphthong']
            elif  lang.lower().startswith('xit'):
                target_col_names = ['consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant',
                                    'trill', 'nasal', 'voice', 'spread_glottis', 'constricted_glottis', 'labial', 'round',
                                    'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal',
                                    'high', 'low', 'front', 'back', 'tense', 'implosive']
            else:
                target_col_names = [class_type]
        else:
            target_col_names = [class_type]

        out += target_col_names

    return out


def probe(
        segment_table,
        class_types,
        lang=None,
        classifier_type='mlp',
        regularization_scale=0.001,
        max_depth=None,
        min_impurity_decrease=0.,
        n_estimators=100,
        n_folds=2,
        units=100,
        compare_to_baseline=False,
        dump_images=False,
        verbose=False,
        name='probe',
        outdir='./probe/'
):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    target_col_names = get_target_cols(class_types, lang=lang)

    X = segment_table
    input_col_names = [c for c in X.columns if is_embedding_dimension.match(c)]
    target_col_names_cur = []
    df_cols = set(X.columns)
    for target_col in target_col_names:
        if target_col in df_cols:
            target_col_names_cur.append(target_col)
        else:
            sys.stderr.write('Ignoring unrecognized target column "%s"...\n' % target_col)
            sys.stderr.flush()

    precision = {}
    recall = {}
    f1 = {}
    accuracy = {}

    precision_baseline = {}
    recall_baseline = {}
    f1_baseline = {}
    accuracy_baseline = {}

    out_dict = {}

    if len(target_col_names_cur):
        for target_col in target_col_names_cur:
            if verbose:
                stderr('  Variable: "%s"\n' % target_col)

            X_cur = X[(~X[target_col].isnull()) & (~X[target_col].isin(['SIL', 'SPN']))]
            fold_size = math.ceil(float(len(X_cur)) / n_folds)
            perm, perm_inv = get_random_permutation(len(X_cur))
            y = X_cur[target_col]
            if len(y.unique()) > 2:
                avg_method = 'macro'
            else:
                avg_method = 'binary'
                if y.sum() > len(y) / 2: # Majority class is positive, flip
                    y = 1 - y

            label_set, label_counts = np.unique(y.values, return_counts=True)
            label_probs = label_counts / label_counts.sum()

            if verbose:
                sys.stderr.write('\r    Label proportions:\n')
                sys.stderr.flush()
                for level, prob in zip(label_set, label_probs):
                    sys.stderr.write('\r      %s: %s\n' % (level, prob))
                    sys.stderr.flush()

            predictions = []
            gold = []

            for j in range(0, len(X_cur), fold_size):
                if verbose:
                    sys.stderr.write(
                        '\r    Fold %d/%d...' % (int(j / fold_size) + 1, math.ceil(len(X_cur) / fold_size)))
                    sys.stderr.flush()
                if classifier_type.lower() == 'random_forest':
                    classifier = RandomForestClassifier(
                        n_estimators=n_estimators,
                        criterion='entropy',
                        class_weight='balanced',
                        max_depth=max_depth,
                        min_impurity_decrease=min_impurity_decrease
                    )
                elif classifier_type.lower() in ['mlr', 'logreg', 'logistic_regression']:
                    classifier = LogisticRegression(
                        class_weight='balanced',
                        C=regularization_scale,
                        solver='lbfgs',
                        multi_class='auto',
                        max_iter=100
                    )
                elif classifier_type.lower() in ['mlp', 'neural_network']:
                    if isinstance(units, str):
                        units = [int(x) for x in units.split()]
                    if not (isinstance(units, list) or isinstance(units, tuple)):
                        units = [int(units)]
                    classifier = MLPClassifier(
                        units,
                        alpha=regularization_scale
                    )

                train_select = np.ones(len(X_cur)).astype('bool')
                train_select[j:j + fold_size] = False
                cv_select = np.logical_not(train_select)
                train_select = train_select[perm_inv]

                X_train = X_cur[input_col_names][train_select]
                y_train = y[train_select]
                if len(y_train.unique()) < 2:
                    break
                X_cv = X_cur[input_col_names][cv_select]
                y_cv = y[cv_select]

                classifier.fit(X_train, y_train)
                predictions.append(classifier.predict(X_cv))
                gold.append(y_cv)

            if len(predictions) > 0:
                predictions = np.concatenate(predictions, axis=0)
                gold = np.concatenate(gold, axis=0)
                precision[target_col] = precision_score(gold, predictions, average=avg_method)
                recall[target_col] = recall_score(gold, predictions, average=avg_method)
                f1[target_col] = f1_score(gold, predictions, average=avg_method)
                accuracy[target_col] = accuracy_score(gold, predictions)

                if verbose:
                    stderr('\n    Cross-validation F1 for variable "%s": %.4f\n' % (target_col, f1[target_col]))

                if compare_to_baseline:
                    predictions_baseline = np.random.choice(label_set, size=(len(gold),), p=label_probs)
                    precision_baseline[target_col] = precision_score(gold, predictions_baseline, average=avg_method)
                    recall_baseline[target_col] = recall_score(gold, predictions_baseline, average=avg_method)
                    f1_baseline[target_col] = f1_score(gold, predictions_baseline, average=avg_method)
                    accuracy_baseline[target_col] = accuracy_score(gold, predictions_baseline)

                    if verbose:
                        stderr('    Baseline F1 for variable "%s":         %.4f\n' % (target_col, f1_baseline[target_col]))

                if dump_images and classifier_type.lower() == 'random_forest':
                    tree_ix = np.random.randint(n_estimators)

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

                    img_str = '/%s_decision_tree_%s.png'

                    outfile = outdir + img_str % (name, target_col)
                    graph.write_png(outfile)

        macro_avg = {
            'precision': sum(precision[x] for x in precision) / sum(1 for _ in precision),
            'recall': sum(recall[x] for x in recall) / sum(1 for _ in recall),
            'f1': sum(f1[x] for x in f1) / sum(1 for _ in f1),
            'accuracy': sum(accuracy[x] for x in accuracy) / sum(1 for _ in accuracy)
        }

        if verbose:
            stderr('    Model macro averages:\n')
            stderr('      P:   %.4f\n' % macro_avg['precision'])
            stderr('      R:   %.4f\n' % macro_avg['recall'])
            stderr('      F1:  %.4f\n' % macro_avg['f1'])
            stderr('      ACC: %.4f\n' % macro_avg['accuracy'])

        if compare_to_baseline:
            macro_avg_baseline = {
                'precision': sum(precision_baseline[x] for x in precision_baseline) / sum(
                    1 for _ in precision_baseline),
                'recall': sum(recall_baseline[x] for x in recall_baseline) / sum(1 for _ in recall_baseline),
                'f1': sum(f1_baseline[x] for x in f1_baseline) / sum(1 for _ in f1_baseline),
                'accuracy': sum(accuracy_baseline[x] for x in accuracy_baseline) / sum(
                    1 for _ in accuracy_baseline)
            }

            if verbose:
                stderr('    Baseline macro averages:\n')
                stderr('      P:   %.4f\n' % macro_avg_baseline['precision'])
                stderr('      R:   %.4f\n' % macro_avg_baseline['recall'])
                stderr('      F1:  %.4f\n' % macro_avg_baseline['f1'])
                stderr('      ACC: %.4f\n' % macro_avg_baseline['accuracy'])

        path_str = '/%s_classifier_scores.txt'
        outfile = outdir + path_str % name
        with open(outfile, 'w') as f:
            f.write('feature precision recall f1 accuracy\n')
            for c in sorted(list(f1.keys())):
                f.write('%s %s %s %s %s\n' % (c, precision[c], recall[c], f1[c], accuracy[c]))
            f.write('MACRO %s %s %s %s\n' % (
            macro_avg['precision'], macro_avg['recall'], macro_avg['f1'], macro_avg['accuracy']))

        if compare_to_baseline:
            path_str = '/%s_baseline_scores.txt'
            outfile = outdir + path_str % name
            with open(outfile, 'w') as f:
                f.write('feature precision recall f1 accuracy\n')
                for c in sorted(list(f1.keys())):
                    f.write('%s %s %s %s %s\n' % (
                    c, precision_baseline[c], recall_baseline[c], f1_baseline[c], accuracy_baseline[c]))
                f.write('MACRO %s %s %s %s\n' % (
                macro_avg_baseline['precision'], macro_avg_baseline['recall'], macro_avg_baseline['f1'],
                macro_avg_baseline['accuracy']))

        for c in sorted(list(f1.keys())):
            key_base = '_'.join([name, c])
            out_dict[key_base + '_p'] = precision[c]
            out_dict[key_base + '_r'] = recall[c]
            out_dict[key_base + '_f1'] = f1[c]

            if compare_to_baseline:
                out_dict[key_base + '_baseline_p'] = precision_baseline[c]
                out_dict[key_base + '_baseline_r'] = recall_baseline[c]
                out_dict[key_base + '_baseline_f1'] = f1_baseline[c]

        out_dict['_'.join([name, 'macro_p'])] = macro_avg['precision']
        out_dict['_'.join([name, 'macro_r'])] = macro_avg['recall']
        out_dict['_'.join([name, 'macro_f1'])] = macro_avg['f1']
        out_dict['_'.join([name, 'macro_acc'])] = macro_avg['accuracy']

        if compare_to_baseline:
            out_dict['_'.join([name, 'baseline_macro_p'])] = macro_avg_baseline['precision']
            out_dict['_'.join([name, 'baseline_macro_r'])] = macro_avg_baseline['recall']
            out_dict['_'.join([name, 'baseline_macro_f1'])] = macro_avg_baseline['f1']
            out_dict['_'.join([name, 'baseline_macro_acc'])] = macro_avg_baseline['accuracy']

        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    return out_dict
