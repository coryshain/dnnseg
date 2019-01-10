import sys
import os
import shutil
import time
import numpy as np
import pickle
import gzip
import argparse

sys.setrecursionlimit(2000)

from dnnseg.config import Config
from dnnseg.data import AcousticDataset, score_segmentation
from dnnseg.kwargs import UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess data (even if saved data object exists in the model directory)')
    argparser.add_argument('-r', '--restart', action='store_true', help='Restart training even if model checkpoint exists (this will overwrite existing checkpoint)')
    args = argparser.parse_args()

    p = Config(args.config)

    if not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    t0 = time.time()
    data_name = 'data_f%s_d%s.obj' %(p['n_coef'], p['order'])
    if not args.preprocess and os.path.exists(p.train_data_dir + '/' + data_name):
        sys.stderr.write('Loading saved training data...\n')
        sys.stderr.flush()
        with open(p.train_data_dir + '/' + data_name, 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = AcousticDataset(
            p.train_data_dir,
            n_coef=p['n_coef'],
            order=p['order'],
        )
        if p.save_preprocessed_data:
            sys.stderr.write('Saving preprocessed training data...\n')
            with open(p.train_data_dir + '/' + data_name, 'wb') as f:
                pickle.dump(train_data, f, protocol=2)

    sys.stderr.write('=' * 50 + '\n')
    sys.stderr.write('TRAINING DATA SUMMARY\n\n')
    sys.stderr.write(train_data.summary(indent=2))
    sys.stderr.write('=' * 50 + '\n\n')

    if p.train_data_dir != p.dev_data_dir:
        if not args.preprocess and os.path.exists(p.dev_data_dir + '/' + data_name):
            sys.stderr.write('Loading saved dev data...\n')
            sys.stderr.flush()
            with open(p.dev_data_dir + '/' + data_name, 'rb') as f:
                cv_data = pickle.load(f)
        else:
            cv_data = AcousticDataset(
                p.dev_data_dir,
                n_coef=p['n_coef'],
                order=p['order'],
            )
            if p.save_preprocessed_data:
                sys.stderr.write('Saving preprocessed dev data...\n')
                with open(p.dev_data_dir + '/' + data_name, 'wb') as f:
                    pickle.dump(cv_data, f, protocol=2)

            sys.stderr.write('=' * 50 + '\n')
            sys.stderr.write('CROSS-VALIDATION DATA SUMMARY\n\n')
            sys.stderr.write(cv_data.summary(indent=2))
            sys.stderr.write('=' * 50 + '\n\n')

    else:
        cv_data = None

    t1 = time.time()

    sys.stderr.write('Data loaded in %ds\n\n' %(t1-t0))
    sys.stderr.flush()

    if p['segtype'] == 'rnd':
        train_data.initialize_random_segmentation(7.4153)
        if cv_data is not None:
            cv_data.initialize_random_segmentation(7.4153)

    sys.stderr.write('Initializing encoder-decoder...\n\n')

    if args.restart and os.path.exists(p.outdir + '/tensorboard'):
        shutil.rmtree(p.outdir + '/tensorboard')

    kwargs = {}

    for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

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

    sys.stderr.write('Fitting encoder-decoder...\n\n')

    dnnseg_model.fit(
        train_data,
        cv_data=cv_data,
        n_iter=p['n_iter'],
        ix2label=train_data.ix2label(p['segtype']),
    )
