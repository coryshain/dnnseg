import sys
import os
import time
import numpy as np
import pickle
import gzip
import argparse

from dnnseg.config import Config
from dnnseg.data import AcousticDataset, binary_segments_to_intervals
from dnnseg.model import UnsupervisedWordClassifier

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess data (even if saved data object exists in the model directory)')
    argparser.add_argument('-r', '--restart', action='store_true', help='Restart training even if model checkpoint exists (this will overwrite existing checkpoint)')
    args = argparser.parse_args()

    p = Config(args.config)

    if not p.use_gpu_if_available:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    t0 = time.time()
    if not args.preprocess and os.path.exists(p.outdir + '/data_train.gz'):
        sys.stderr.write('Loading saved training data...\n')
        sys.stderr.flush()
        with gzip.open(p.outdir + '/data_train.gz', 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = AcousticDataset(p.train_data_dir)
        if p.save_preprocessed_data:
            sys.stderr.write('Saving preprocessed training data...\n')
            with gzip.open(p.outdir + '/data_train.gz', 'wb') as f:
                pickle.dump(train_data, f, protocol=2)

    if not args.preprocess and os.path.exists(p.outdir + '/data_dev.gz'):
        sys.stderr.write('Loading saved dev data...\n')
        sys.stderr.flush()
        with gzip.open(p.outdir + '/data_dev.gz', 'rb') as f:
            dev_data = pickle.load(f)
    else:
        dev_data = AcousticDataset(p.dev_data_dir)
        if p.save_preprocessed_data:
            sys.stderr.write('Saving preprocessed dev data...\n')
            with gzip.open(p.outdir + '/data_dev.gz', 'wb') as f:
                pickle.dump(dev_data, f, protocol=2)

    t1 = time.time()

    sys.stderr.write('Data loaded in %ds\n\n' %(t1-t0))
    sys.stderr.flush()

    train_inputs, train_inputs_mask = train_data.words()
    train_targets, train_targets_mask = train_data.targets(segment_type='wrd')
    train_labels = train_data.labels(one_hot=False)

    dev_inputs, dev_inputs_mask = dev_data.words()
    dev_targets, dev_targets_mask = dev_data.targets(segment_type='wrd')
    dev_labels = dev_data.labels(one_hot=False)

    sys.stderr.write('Initializing unsupervised word classifier...\n\n')

    dnnseg_model = UnsupervisedWordClassifier(
        p.k,
        temp=p.temp,
        trainable_temp=p.trainable_temp,
        binary_classifier=p.binary_classifier,
        emb_dim=p.emb_dim,
        encoder_type=p.encoder_type,
        decoder_type=p.decoder_type,
        dense_n_layers=p.dense_n_layers,
        unroll_rnn=p.unroll_rnn,
        conv_n_filters=p.conv_n_filters,
        output_scale = p.output_scale,
        n_coef=p.n_coef,
        n_timesteps=train_inputs.shape[1],
        learning_rate=p.learning_rate
    )

    dnnseg_model.build(len(train_inputs), outdir=p.outdir, restore=not args.restart)


    sys.stderr.write('Fitting unsupervised word classifier...\n\n')

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
        n_iter=p.n_iter
    )
