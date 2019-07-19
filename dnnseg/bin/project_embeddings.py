import sys
import os
import pandas as pd
import argparse

from dnnseg.config import Config
from dnnseg.data import project_matching_segments
from dnnseg.plot import plot_projections

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Project saved embeddings from a DNN-Seg run.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-f', '--feature_names', default=['english'], nargs='+', help='Names of phonological features to use to color plots.')
    args = argparser.parse_args()

    p = Config(args.config)

    if p['label_map_file'] is not None and os.path.exists(p['label_map_file']):
        label_map = pd.read_csv(p['label_map_file'])
        label_map = dict(zip(label_map.source, label_map.target))
    else:
        label_map = None

    if p['feature_map_file'] is not None and os.path.exists(p['feature_map_file']):
        feature_table = pd.read_csv(p['feature_map_file'])
        new_cols = []
        for i in range(len(feature_table.columns)):
            new_cols.append(feature_table.columns[i].replace(' ', '_').replace('+', '-'))
        feature_table.columns = new_cols
        if not 'label' in feature_table.columns and 'symbol' in feature_table.columns:
            feature_table['label'] = feature_table.symbol
    else:
        feature_table = None

    if args.feature_names == ['english']:
        feature_names = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant', 'nasal', 'voice', 'spread_glottis', 'labial', 'round', 'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back', 'tense', 'stress', 'diphthong']
    elif args.feature_names == ['xitsonga']:
        feature_names = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant', 'trill', 'nasal', 'voice', 'spread_glottis', 'constricted_glottis', 'labial', 'round', 'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back', 'tense', 'implosive']
    else:
        feature_names = args.feature_names

    embeddings = [x for x in os.listdir(p.outdir) if x.startswith('matching_')]

    for e in embeddings:
        l = int(e.split('_')[-1].split('.')[0][1:])
        df = pd.read_csv(p.outdir + '/' + e, sep=' ')
        df = project_matching_segments(df)
        plot_projections(
            df,
            label_map=label_map,
            feature_table=feature_table,
            feature_names=feature_names,
            directory=p.outdir,
            prefix='l%d_' % l,
            suffix='.png'
        )


