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
from dnnseg.data import AcousticDataset, score_segs
from dnnseg.kwargs import UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess data (even if saved data object exists in the model directory)')
    argparser.add_argument('-r', '--restart', action='store_true', help='Restart training even if model checkpoint exists (this will overwrite existing checkpoint)')
    argparser.add_argument('-s', '--segtype', type=str, default='wrd', help='Segment type to use for training (one of ["wrd", "phn"]')
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

    if p.train_data_dir == p.dev_data_dir:
        dev_data = train_data
    else:
        if not args.preprocess and os.path.exists(p.dev_data_dir + '/' + data_name):
            sys.stderr.write('Loading saved dev data...\n')
            sys.stderr.flush()
            with open(p.dev_data_dir + '/' + data_name, 'rb') as f:
                dev_data = pickle.load(f)
        else:
            dev_data = AcousticDataset(
                p.dev_data_dir,
                n_coef=p['n_coef'],
                order=p['order'],
            )
            if p.save_preprocessed_data:
                sys.stderr.write('Saving preprocessed dev data...\n')
                with open(p.dev_data_dir + '/' + data_name, 'wb') as f:
                    pickle.dump(dev_data, f, protocol=2)

    t1 = time.time()

    sys.stderr.write('Data loaded in %ds\n\n' %(t1-t0))
    sys.stderr.flush()

    sys.stderr.write('=' * 50 + '\n')
    sys.stderr.write('TRAINING DATA SUMMARY\n\n')
    sys.stderr.write(train_data.summary(indent=2))
    sys.stderr.write('=' * 50 + '\n\n')

    sys.stderr.write('=' * 50 + '\n')
    sys.stderr.write('CROSS-VALIDATION DATA SUMMARY\n\n')
    sys.stderr.write(dev_data.summary(indent=2))
    sys.stderr.write('=' * 50 + '\n\n')

    if args.segtype == 'rnd':
        train_data.initialize_random_segmentation(7.4153)

    # print(score_segs(train_data.segments('phn'), train_data.segments('phn')))

    if p['pad_seqs']:
        if p['mask_padding'] and ('softhmlstm' in p['encoder_type'].lower() or 'rnn' in p['encoder_type'].lower()):
            input_padding = 'post'
        else:
            input_padding = 'pre'
        target_padding = 'post'
    else:
        input_padding = None
        target_padding = None

    print('Input padding')
    print(input_padding)
    print('Target padding')
    print(target_padding)
    print()

    sys.stderr.write('Extracting training and cross-validation data...\n')
    t0 = time.time()
    train_inputs, train_inputs_mask = train_data.inputs(
        segmentation_type=args.segtype,
        padding=input_padding,
        max_len=p['max_len'],
        normalize=p['normalize_data'],
        center=p['center_data'],
        resample=p['resample_inputs']
    )
    train_targets, train_targets_mask = train_data.targets(
        segmentation_type=args.segtype,
        padding=target_padding,
        max_len=p['max_len'],
        reverse=p['reverse_targets'],
        normalize=p['normalize_data'],
        center=p['center_data'],
        resample=p['resample_outputs']
    )
    train_labels = train_data.labels(one_hot=False, segment_type=args.segtype)

    if p.train_data_dir == p.dev_data_dir:
        dev_inputs = train_inputs
        dev_inputs_mask = train_inputs_mask
        dev_targets = train_targets
        dev_targets_mask = train_targets_mask
        dev_labels = train_labels
    else:
        dev_inputs, dev_inputs_mask = dev_data.inputs(
            segmentation_type=args.segtype,
            padding=input_padding,
            max_len=p['max_len'],
            normalize=p['normalize_data'],
            center=p['center_data'],
            resample=p['resample_inputs']
        )
        dev_targets, dev_targets_mask = dev_data.targets(
            segmentation_type=args.segtype,
            padding=input_padding,
            max_len=p['max_len'],
            reverse=p['reverse_targets'],
            normalize=p['normalize_data'],
            center=p['center_data'],
            resample=p['resample_outputs']
        )
        dev_labels = train_data.labels(one_hot=False, segment_type=args.segtype)

    t1 = time.time()

    sys.stderr.write('Training and cross-validation data extracted in %ds\n\n' %(t1-t0))
    sys.stderr.flush()


    sys.stderr.write('Initializing encoder-decoder...\n\n')

    if args.restart and os.path.exists(p.outdir + '/tensorboard'):
        shutil.rmtree(p.outdir + '/tensorboard')

    kwargs = {}

    for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    if p['pad_seqs']:
        kwargs['n_timesteps_input'] = train_inputs.shape[1]
        kwargs['n_timesteps_output'] = train_targets.shape[1]
    else:
        if p['resample_inputs']:
            kwargs['n_timesteps_input'] = train_inputs[0].shape[1]
        if p['resample_outputs']:
            kwargs['n_timesteps_output'] = train_targets[0].shape[1]

    if p['network_type'] == 'mle':
        from dnnseg.model import AcousticEncoderDecoderMLE

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = AcousticEncoderDecoderMLE(
            p['k'],
            **kwargs
        )
    else:
        from dnnseg.model_bayes import AcousticEncoderDecoderBayes

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = AcousticEncoderDecoderBayes(
            p['k'],
            **kwargs
        )

    dnnseg_model.build(len(train_inputs), outdir=p.outdir, restore=not args.restart)


    sys.stderr.write('Fitting encoder-decoder...\n\n')

    dnnseg_model.fit(
        train_inputs,
        train_inputs_mask,
        train_targets,
        train_targets_mask,
        train_labels,
        X_cv=dev_inputs,
        X_mask_cv=dev_inputs_mask,
        y_cv=dev_targets,
        y_mask_cv=dev_targets_mask,
        labels_cv=dev_labels,
        n_iter=p['n_iter'],
        ix2label=train_data.ix2label(args.segtype),
    )
