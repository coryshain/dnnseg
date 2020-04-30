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
from dnnseg.util import load_dnnseg, stderr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', nargs='+', help='Path(s) to configuration file.')
    argparser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess data (even if saved data object exists in the model directory)')
    argparser.add_argument('-c', '--force_cpu', action='store_true', help='Do not use GPU. If not specified, GPU usage defaults to the value of the **use_gpu_if_available** configuration parameter.')
    args = argparser.parse_args()

    outputs = {}
    val_data = None

    for config in args.config:
        p = Config(config)

        if args.force_cpu or not p['use_gpu_if_available']:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        t0 = time.time()

        if p['data_type'].lower() == 'acoustic':
            data_name = 'data_%s_f%s_d%s.obj' %(p['filter_type'], p['n_coef'], p['order'])
        else:
            data_name = 'data.obj'

        preprocessed = True

        if val_data is None:
            train_data_dirs = p.train_data_dir.split()

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

            t1 = time.time()

            stderr('Data loaded in %ds\n\n' %(t1-t0))
            sys.stderr.flush()

            if p['segtype'] == 'rnd':
                val_data.initialize_random_segmentation(7.4153)

            data = val_data

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
                train_data=val_data,
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

        stderr('Loading DNNSeg...\n\n')

        kwargs = {}

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = load_dnnseg(p['outdir'])

        outputs_cur = dnnseg_model.get_segmentation_arrays(
            val_data,
            whole_file=True,
            verbose=True
        )

        key = p['outdir']

        if key in outputs:
            i = 1
            key += '_%d' % i
            while key in outputs:
                i += 1
                key += '_%d' % i

        outputs[key] = outputs_cur

    segmentations = None
    phn_boundaries = None
    phn_labels = None
    wrd_boundaries = None
    wrd_labels = None
    n = 0
    for n, o in enumerate(outputs):
        if segmentations is None:
            segmentations = outputs[o]['segmentations']
        else:
            for i in range(len(segmentations)):
                for j in range(len(segmentations[i])):
                    segmentations[i][j] = (segmentations[i][j] * n + outputs[o]['segmentations'][i][j]) / (n + 1)
        if phn_boundaries is None:
            phn_boundaries = outputs[o]['phn_boundaries']
        if phn_labels is None:
            phn_labels = outputs[o]['phn_labels']
        if wrd_boundaries is None:
            wrd_boundaries = outputs[o]['wrd_boundaries']
        if wrd_labels is None:
            wrd_labels = outputs[o]['wrd_labels']

    for i in range(len(segmentations)):
        for j in range(len(segmentations[i])):
            segmentations[i][j] = (segmentations[i][j] > 0.5).astype(float)

    scores, pred_tables, phn_tables, wrd_tables, summary = dnnseg_model.get_segmentation_eval(
            val_data,
            segtype=dnnseg_model.segtype,
            padding=None,
            segmentations=segmentations,
            states=None,
            phn_boundaries=phn_boundaries,
            phn_labels=phn_labels,
            wrd_boundaries=wrd_boundaries,
            wrd_labels=wrd_labels,
            random_baseline=True,
            save_embeddings=True,
            report_classeval=False,
            plot=False,
            verbose=True
    )

    stderr(summary)
