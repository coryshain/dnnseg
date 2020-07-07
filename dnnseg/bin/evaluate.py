import sys
import os
import time
import pandas as pd
import argparse
import yaml

sys.setrecursionlimit(2000)

from dnnseg.config import Config
from dnnseg.data import Dataset, cache_data
from dnnseg.probe import probe
from dnnseg.util import stderr, sn, load_dnnseg


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Evaluate a trained DNNSeg model.
    ''')
    argparser.add_argument('configs', nargs='+', help='Path(s) to configuration file(s).')
    argparser.add_argument('-d', '--data', nargs='+', default='val', help='Data to use for evaluation (space-delimited). Either one of ``["train", "val", "test"]``, or a path to a directory containing evaluation data. Each argument to this option is treated as a separate evaluation set.')
    argparser.add_argument('-N', '--names', nargs='*', help='Names for evaluation data. If empty, names are inferred. Otherwise, must have the same number of elements as ``data``.')
    argparser.add_argument('-e', '--eval_measures', nargs='+', default=['all'], help='Measures to output. One of ``["all", "segmentation", "classification", "objective", "label_probe", "feature_probe"]``. ``"segmentation"`` runs boundary and word segmentation scores for words and (if relevant) phones. ``"classification"`` creates hard labels by rounding embeddings and performs unsupervised clustering evaluation for word and (if relevant) phone categories. ``"objective"`` computes the mean value of each of the loss components over the dataset, without updating the model. ``"label_probe"`` runs a supervised MLP probe for phone labels if relevant, otherwise skipped. ``"feature_probe"`` runs a supervised MLP probe for features if relevant, otherwise skipped. ``"all"`` runs all measures.')
    argparser.add_argument('-t', '--classifier_type', default='mlp',help='Type of classifier to use. One of ``["logreg", "mlp", "random_forest"]``.')
    argparser.add_argument('-r', '--regularization_scale', type=float, default=0.001, help='Level of regularization to use in MLR classification.')
    argparser.add_argument('-M', '--max_depth', type=float, default=None, help='Maximum permissible tree depth.')
    argparser.add_argument('-m', '--min_impurity_decrease', type=float, default=0., help='Minimum impurity decrease necessary to make a split.')
    argparser.add_argument('-n', '--n_estimators', type=int, default=100, help='Number of estimators (trees) in random forest.')
    argparser.add_argument('-f', '--n_folds', type=int, default=10, help='Number of folds in cross-validation (>= 2).')
    argparser.add_argument('-u', '--units', default=[100], nargs='+', help='Space-delimited list of integers representing number of hidden units to use in MLP classifier.')
    argparser.add_argument('-b', '--compare_to_baseline', action='store_true', help='Whether to compute scores for a matched random baseline.')
    argparser.add_argument('-i', '--images', action='store_true', help='Dump representative images of decision trees.')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Write progress reports to stderr.')
    argparser.add_argument('-c', '--force_cpu', action='store_true', help='Do not use GPU. If not specified, GPU usage defaults to the value of the **use_gpu_if_available** configuration parameter.')
    args = argparser.parse_args()

    measures = []
    for m in args.eval_measures:
        if m.lower() == 'all':
            measures = ["all", "segmentation", "classification", "objective", "label_probe", "feature_probe"]
            break
        measures.append(m.lower())

    for c in args.configs:
        p = Config(c)

        if args.force_cpu or not p['use_gpu_if_available']:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        t0 = time.time()

        if p['data_type'].lower() == 'acoustic':
            data_name = 'data_%s_f%s_d%s.obj' %(p['filter_type'], p['n_coef'], p['order'])
        else:
            data_name = 'data.obj'

        if 'objective' in measures:
            train_data_dirs = p.train_data_dir.split()
            datasets = args.data

            if 'objective' in measures:
                data_dirs = p.train_data_dir.split(';')
                train_data = Dataset(
                    data_dirs,
                    datatype=p['data_type'].lower(),
                    filter_type=p['filter_type'].lower(),
                    n_coef=p['n_coef'],
                    order=p['order'],
                    force_preprocess=False,
                    save_preprocessed_data=p.save_preprocessed_data
                )
                if p['oracle_boundaries']:
                    for x in p['oracle_boundaries'].split():
                        if x.startswith('rnd'):
                            length = float(x[3:])
                            train_data.initialize_random_segmentation(length, save=True)

                if args.verbose:
                    stderr('=' * 50 + '\n')
                    stderr('TRAINING DATA SUMMARY\n\n')
                    stderr(train_data.summary(indent=2))
                    stderr('=' * 50 + '\n\n')
            else:
                train_data = None

        for i, dataset in enumerate(args.data):
            info_dict = {}

            if args.names:
                name = args.names[i]
            else:
                name = sn(dataset)

            if dataset.lower() == 'train':
                dataset = p.train_data_dir
            elif dataset.lower() == 'val':
                dataset = p.val_data_dir
            elif dataset.lower() == 'test':
                dataset = p.test_data_dir

            data_dirs = dataset.split(';')
            data = Dataset(
                data_dirs,
                datatype=p['data_type'].lower(),
                filter_type=p['filter_type'].lower(),
                n_coef=p['n_coef'],
                order=p['order'],
                force_preprocess=False,
                save_preprocessed_data=p.save_preprocessed_data
            )
            if p['oracle_boundaries']:
                for x in p['oracle_boundaries'].split():
                    if x.startswith('rnd'):
                        length = float(x[3:])
                        data.initialize_random_segmentation(length, save=True)

            if args.verbose:
                stderr('=' * 50 + '\n')
                stderr('DATA SUMMARY\n\n')
                stderr(data.summary(indent=2))
                stderr('=' * 50 + '\n\n')

            t1 = time.time()

            stderr('Data loaded in %ds\n\n' %(t1-t0))
            sys.stderr.flush()

            stderr('Caching data...\n')
            sys.stderr.flush()

            if p['pad_seqs']:
                if p['mask_padding'] and ('hmlstm' in p['encoder_type'].lower() or 'rnn' in p['encoder_type'].lower()):
                    input_padding = 'post'
                else:
                    input_padding = 'pre'
                target_padding = 'post'
            else:
                input_padding = None
                target_padding = None

            cache_data(
                other_data=data,
                streaming=p['streaming'],
                max_len=p['max_len'],
                window_len_bwd=p['window_len_bwd'],
                window_len_fwd=p['window_len_fwd'],
                segtype=p['segtype'],
                data_normalization=p['data_normalization'],
                reduction_axis=p['reduction_axis'],
                use_normalization_mask=p['use_normalization_mask'],
                predict_deltas=p['predict_deltas'],
                input_padding=input_padding,
                target_padding=target_padding,
                reverse_targets=p['reverse_targets'],
                resample_inputs=p['resample_inputs'],
                resample_targets_bwd=p['resample_targets_bwd'],
                resample_targets_fwd=p['resample_targets_fwd'],
                task=p['task'],
                data_type=p['data_type']
            )

            m = load_dnnseg(p['outdir'])

            if 'classification' in measures or 'segmentation' in measures or 'label_probe' in measures or 'feature_probe' in measures:
                eval_dict = m.run_evaluation(
                    data,
                    n_plot=None,
                    ix2label=data.ix2label(p['segtype']),
                    training=False,
                    segtype=p['segtype'],
                    random_baseline=True,
                    evaluate_classifier='classification' in measures,
                    evaluate_segmenter='segmentation' in measures,
                    save_embeddings=True,
                    verbose=args.verbose
                )

                if 'classification_scores' in eval_dict:
                    for l, layer in enumerate(eval_dict['classification_scores']):
                        for targ_seg_level in layer:
                            for sel in layer[targ_seg_level]:
                                for system in layer[targ_seg_level][sel]:
                                    for metric in layer[targ_seg_level][sel][system]:
                                        info_dict['_'.join([targ_seg_level, sel, system, metric, name, 'l%d' % l])] = float(layer[targ_seg_level][sel][system][metric])

                if 'segmentation_scores' in eval_dict:
                    for l, layer in enumerate(eval_dict['segmentation_scores']):
                        for seg_gold in layer:
                            for metric in layer[seg_gold]:
                                info_dict['_'.join([seg_gold, metric, name, 'l%d' % l])] = float(layer[seg_gold][metric])

            if 'objective' in measures:
                objective_dict = m.get_loss(
                    train_data,
                    verbose=args.verbose
                )

                for x in objective_dict:
                    val = objective_dict[x]
                    if isinstance(val, list):
                        for l, v in enumerate(val):
                            info_dict['_'.join([x, 'l%d' % l, name])] = float(val[l])
                    else:
                        info_dict['_'.join([x, name])] = float(val)

            if 'label_probe' in measures or 'feature_probe' in measures:
                analysis_dir = os.path.join(p['outdir'], 'probe')

                for path in os.listdir(p['outdir'] + '/tables/'):
                    if path.startswith('embeddings') and name in path and path.endswith('.csv'):
                        if args.verbose:
                            sys.stderr.write('*' * 50 + '\n')
                            sys.stderr.write('Analyzing segment table %s...\n' % path)
                            sys.stderr.flush()

                        df = pd.read_csv(os.path.join(p['outdir'] + '/tables/', path), sep=' ')

                        if p['feature_map_file'] is not None and p['feature_map_file'].lower().startswith('eng'):
                            lang = 'english'
                        elif p['feature_map_file'] is not None and p['feature_map_file'].lower().startswith('xit'):
                            lang = 'xitsonga'
                        else:
                            lang = None

                        if 'label_probe' in measures:
                            label_probe_dict = probe(
                                df,
                                'phn_label',
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

                            if len(label_probe_dict):
                                for x in label_probe_dict:
                                    key = x
                                    if 'macro' in key:
                                        key = key.replace('macro', 'phn_label_macro')
                                    info_dict[key] = float(label_probe_dict[x])

                        if 'feature_probe' in measures:
                            feature_probe_dict = probe(
                                df,
                                'features',
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

                            if len(feature_probe_dict):
                                for x in feature_probe_dict:
                                    key = x
                                    if 'macro' in key:
                                        key = key.replace('macro', 'phn_label_macro')
                                    info_dict[key] = float(feature_probe_dict[x])

            info_dict['num_iter'] = int(m.global_step.eval(session=m.sess))
            info_dict['model_path'] = m.outdir

            out = yaml.dump(info_dict)

            with open(m.outdir + '/eval_table_%s.yml' % name, 'w') as f:
                f.write(out)


