import sys
import os
import shutil
import time
import pickle
import numpy as np
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

    if not args.preprocess and os.path.exists(p.train_data_dir + '/' + data_name):
        stderr('Loading saved training data...\n')
        sys.stderr.flush()
        with open(p.train_data_dir + '/' + data_name, 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = Dataset(
            p.train_data_dir,
            datatype=p['data_type'].lower(),
            filter_type=p['filter_type'].lower(),
            n_coef=p['n_coef'],
            order=p['order'],
        )
        if p.save_preprocessed_data:
            stderr('Saving preprocessed training data...\n')
            with open(p.train_data_dir + '/' + data_name, 'wb') as f:
                pickle.dump(train_data, f, protocol=2)
    if p['oracle_boundaries'] and p['oracle_boundaries'].lower() == 'rnd':
        assert p['random_oracle_segmentation_rate'], 'random_oracle_segmentation_rate must be provided when oracle_boundaries=="rnd".'
        train_data.initialize_random_segmentation(p['random_oracle_segmentation_rate'])

    stderr('=' * 50 + '\n')
    stderr('TRAINING DATA SUMMARY\n\n')
    stderr(train_data.summary(indent=2))
    stderr('=' * 50 + '\n\n')

    if p.train_data_dir != p.val_data_dir:
        if not args.preprocess and os.path.exists(p.val_data_dir + '/' + data_name):
            stderr('Loading saved validation data...\n')
            sys.stderr.flush()
            with open(p.val_data_dir + '/' + data_name, 'rb') as f:
                val_data = pickle.load(f)
        else:
            val_data = Dataset(
                p.val_data_dir,
                datatype=p['data_type'].lower(),
                filter_type=p['filter_type'].lower(),
                n_coef=p['n_coef'],
                order=p['order'],
            )
            if p.save_preprocessed_data:
                stderr('Saving preprocessed dev data...\n')
                with open(p.val_data_dir + '/' + data_name, 'wb') as f:
                    pickle.dump(val_data, f, protocol=2)

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
        predict_deltas=p['predict_deltas'],
        input_padding=input_padding,
        target_padding=target_padding,
        reverse_targets=p['reverse_targets'],
        resample_inputs=p['resample_inputs'],
        resample_targets_bwd=p['resample_targets_bwd'],
        resample_targets_fwd=p['resample_targets_fwd'],
        oracle_boundaries=p['oracle_boundaries'],
        task=p['task'],
        data_type=p['data_type']
    )

    # data_feed = train_data.get_data_feed('val_files', minibatch_size=1, randomize=False)
    # for batch in data_feed:
    #     print(np.stack([batch['oracle_boundaries'][0,:,0],batch['oracle_labels'][0,:,0]], axis=1)[:100])
    #     input()

    stderr('Initializing encoder-decoder...\n\n')

    if args.restart and os.path.exists(p.outdir + '/tensorboard'):
        shutil.rmtree(p.outdir + '/tensorboard')

    kwargs = {}

    for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    if p['data_type'] == 'text':
        kwargs['n_coef'] = len(train_data.ix2char)
        kwargs['predict_deltas'] = False

    if p['network_type'] == 'mle':
        from dnnseg.model import AcousticEncoderDecoderMLE

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = AcousticEncoderDecoderMLE(
            p['k'],
            train_data,
            **kwargs
        )
    else:
        from dnnseg.model_bayes import AcousticEncoderDecoderBayes

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = AcousticEncoderDecoderBayes(
            p['k'],
            train_data,
            **kwargs
        )

    dnnseg_model.build(len(train_data.segments(segment_type=p['segtype'])), outdir=p.outdir, restore=not args.restart)

    stderr('Fitting encoder-decoder...\n\n')

    dnnseg_model.fit(
        train_data,
        val_data=val_data,
        n_iter=p['n_iter'],
        ix2label=train_data.ix2label(p['segtype']),
    )
