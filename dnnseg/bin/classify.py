import sys
import os
import time
import pickle
import pandas as pd
import argparse

from dnnseg.config import Config
from dnnseg.util import load_dnnseg

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-p', '--partition', default='train', help='Name of partition from which to extract classifications (one of ["train", "dev", "test"]')
    argparser.add_argument('-s', '--segtype', type=str, default='phn', help='Segment type to use for training (one of ["vad", "wrd", "phn"]')
    argparser.add_argument('-g', '--goldfeats', type=str, default=None, help='Path to gold features file to merge with classification table. If unspecified, no gold features columns will be added.')
    args = argparser.parse_args()

    p = Config(args.config)

    if not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    t0 = time.time()
    data_name = 'data_f%s_d%s.obj' %(p['n_coef'], p['order'])

    sys.stderr.write('Loading saved training data...\n')
    sys.stderr.flush()

    data_map = {
        'train': p.train_data_dir,
        'dev': p.dev_data_dir,
        'cv': p.dev_data_dir,
        'test': p.test_data_dir
    }

    with open(data_map[args.partition] + '/' + data_name, 'rb') as f:
        data = pickle.load(f)

    dnnseg_model = load_dnnseg(p.outdir)

    segments, _, summary = dnnseg_model.classify_utterances(
        data,
        segtype=args.segtype,
        ix2label=data.ix2label(args.segtype)
    )

    if args.goldfeats:
        gold_feature_map = pd.read_csv(args.goldfeats)
        new_cols = []
        for i in range(len(gold_feature_map.columns)):
            new_cols.append(gold_feature_map.columns[i].replace(' ', '_').replace('+', '-'))
        gold_feature_map.columns = new_cols
        if not 'label' in gold_feature_map.columns and 'symbol' in gold_feature_map.columns:
            gold_feature_map['label'] = gold_feature_map.symbol
        segments = segments.merge(gold_feature_map, on=['label'])

    outfile = p.outdir + '/' + 'classifications_%s' % args.partition + '.csv'
    segments.to_csv(outfile, na_rep='nan', index=False)

    outfile = p.outdir + '/' + 'classification_scores_%s' % args.partition + '.txt'
    with open(outfile, 'w') as f:
        f.write(summary)




