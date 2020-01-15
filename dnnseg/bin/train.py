import sys
import os
import shutil
import time
import numpy as np
import pandas as pd
import argparse

sys.setrecursionlimit(2000)

from dnnseg.config import Config
from dnnseg.data import Dataset, cache_data
from dnnseg.kwargs import UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS
from dnnseg.util import stderr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess data (even if saved data object exists in the model directory)')
    argparser.add_argument('-r', '--restart', action='store_true', help='Restart training even if model checkpoint exists (this will overwrite existing checkpoint)')
    argparser.add_argument('-c', '--force_cpu', action='store_true', help='Do not use GPU. If not specified, GPU usage defaults to the value of the **use_gpu_if_available** configuration parameter.')
    args = argparser.parse_args()

    p = Config(args.config)

    if args.force_cpu or not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    t0 = time.time()

    if p['data_type'].lower() == 'acoustic':
        data_name = 'data_%s_f%s_d%s.obj' %(p['filter_type'], p['n_coef'], p['order'])
    else:
        data_name = 'data.obj'

    preprocessed = True
    train_data_dirs = p.train_data_dir.split()

    data_dirs = p.train_data_dir.split(';')
    train_data = Dataset(
        data_dirs,
        datatype=p['data_type'].lower(),
        filter_type=p['filter_type'].lower(),
        n_coef=p['n_coef'],
        order=p['order'],
        force_preprocess=args.preprocess,
        save_preprocessed_data=p.save_preprocessed_data
    )
    if p['oracle_boundaries'] and p['oracle_boundaries'].lower() == 'rnd':
        assert p['random_oracle_segmentation_rate'], 'random_oracle_segmentation_rate must be provided when oracle_boundaries=="rnd".'
        train_data.initialize_random_segmentation(p['random_oracle_segmentation_rate'])

    stderr('=' * 50 + '\n')
    stderr('TRAINING DATA SUMMARY\n\n')
    stderr(train_data.summary(indent=2))
    stderr('=' * 50 + '\n\n')

    if p.train_data_dir != p.val_data_dir:
        data_dirs = p.val_data_dir.split(';')
        val_data = Dataset(
            data_dirs,
            datatype=p['data_type'].lower(),
            filter_type=p['filter_type'].lower(),
            n_coef=p['n_coef'],
            order=p['order'],
            force_preprocess=args.preprocess,
            save_preprocessed_data=p.save_preprocessed_data
        )

        stderr('=' * 50 + '\n')
        stderr('VALIDATION DATA SUMMARY\n\n')
        stderr(val_data.summary(indent=2))
        stderr('=' * 50 + '\n\n')

    else:
        val_data = None

    t1 = time.time()

    stderr('Data loaded in %ds\n\n' %(t1-t0))
    sys.stderr.flush()

    if p['segtype'] == 'rnd':
        train_data.initialize_random_segmentation(7.4153)
        if val_data is not None:
            val_data.initialize_random_segmentation(7.4153)

    data = train_data

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
        train_data=train_data,
        val_data=val_data,
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

    if False:
        layers_encoder = len(p['n_units_encoder'].split())

        feats, splits, _ = data.features(
            mask='vad'
        )

        segmentations = [[] for _ in range(layers_encoder - 1)]
        states = [[] for _ in range(layers_encoder - 1)]
        for f, s in zip(feats, splits):
            for l in range(layers_encoder - 1):
                segmentations_cur = [(np.random.random((1, f.shape[1])) > 0.8).astype(float) for _ in
                                     range(layers_encoder - 1)]
                states_cur = [(np.random.random((1, f.shape[1], 7)) > 0.5).astype(float) for _ in
                              range(layers_encoder - 1)]
                splits_cur = np.where(np.concatenate([np.zeros((1,)), s[0, :-1]]))[0]

                segmentations[l].append(segmentations_cur[l][0])
                states[l].append(states_cur[l][0])

        # segmentations = [[np.squeeze(x) for x in data.cache['files']['phn_boundaries']]] * (layers_encoder - 1)

        data_feed = data.get_data_feed('files', minibatch_size=128, randomize=False)
        phn_boundaries = []
        phn_labels = []
        wrd_boundaries = []
        wrd_labels = []

        for file in data_feed:
            X_batch = file['X']
            fixed_boundaries_batch = file['fixed_boundaries']
            phn_boundaries_batch = np.squeeze(file['phn_boundaries'])
            phn_boundaries.append(phn_boundaries_batch)
            phn_labels_batch = np.squeeze(file['phn_labels'])
            phn_labels.append(phn_labels_batch)
            wrd_boundaries_batch = np.squeeze(file['wrd_boundaries'])
            wrd_boundaries.append(wrd_boundaries_batch)
            wrd_labels_batch = np.squeeze(file['wrd_labels'])
            wrd_labels.append(wrd_labels_batch)

        pred_tables = data.get_segment_tables(
            segmentations=segmentations,
            parent_segment_type='vad',
            states=states,
            phn_labels=phn_labels,
            wrd_labels=wrd_labels,
            state_activation='sigmoid',
            smoothing_algorithm=None,
            smoothing_algorithm_params=None,
            n_points=None,
            padding=None
        )

        phn_tables = data.get_segment_tables(
            segmentations=[phn_boundaries] * (layers_encoder - 1),
            states=states,
            phn_labels=phn_labels,
            wrd_labels=None,
            parent_segment_type='vad',
            state_activation='sigmoid',
            smoothing_algorithm=None,
            smoothing_algorithm_params=None,
            n_points=None,
            padding=None
        )

        if os.path.exists(p['feature_map_file']):
            feature_map = pd.read_csv(p['feature_map_file'])
        if os.path.exists(p['label_map_file']):
            label_map = pd.read_csv(p['label_map_file'])

        matched_phn = data.extract_matching_segment_embeddings(
            'phn',
            pred_tables[0],
            tol=0.02
        )

        wrd_tables = data.get_segment_tables(
            segmentations=[wrd_boundaries] * (layers_encoder - 1),
            states=states,
            phn_labels=None,
            wrd_labels=wrd_labels,
            parent_segment_type='vad',
            state_activation='sigmoid',
            smoothing_algorithm=None,
            smoothing_algorithm_params=None,
            n_points=None,
            padding=None
        )

        pred_tables[l].to_csv(
            p['outdir'] + '/embeddings_pred_segs_l%d.csv' % l,
            sep=' ',
            index=False
        )
        phn_tables[l].to_csv(
            p['outdir'] + '/embeddings_gold_phn_segs_l%d.csv' % l,
            sep=' ',
            index=False
        )
        wrd_tables[l].to_csv(
            p['outdir'] + '/embeddings_gold_wrd_segs_l%d.csv' % l,
            sep=' ',
            index=False
        )

        exit()


    stderr('Initializing DNNSeg...\n\n')

    if args.restart and os.path.exists(p.outdir + '/tensorboard'):
        shutil.rmtree(p.outdir + '/tensorboard')

    kwargs = {}

    for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    if p['data_type'] == 'text':
        kwargs['n_coef'] = len(train_data.ix2char)
        kwargs['predict_deltas'] = False

    if p['network_type'] == 'mle':
        from dnnseg.model import DNNSegMLE

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = DNNSegMLE(
            train_data,
            **kwargs
        )
    else:
        from dnnseg.model_bayes import DNNSegBayes

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = DNNSegBayes(
            p['k'],
            train_data,
            **kwargs
        )

    dnnseg_model.build(len(train_data.segments(segment_type=p['segtype'])), outdir=p.outdir, restore=not args.restart)

    stderr('Fitting DNNSeg...\n\n')

    dnnseg_model.fit(
        train_data,
        val_data=val_data,
        n_iter=p['n_iter'],
        ix2label=train_data.ix2label(p['segtype']),
    )
