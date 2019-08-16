import sys
import os
import re
import pandas as pd
import argparse

from dnnseg.config import Config
from dnnseg.data import project_matching_segments
from dnnseg.plot import plot_projections


emb_file = re.compile('embeddings_(.+)_segs_l([0-9]+).csv')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Project saved embeddings from a DNN-Seg run.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-f', '--feature_names', default=['english'], nargs='+', help='Names of phonological features to use to color plots.')
    argparser.add_argument('-m', '--method', default='mds', help='Embedding method to use for projections. One of ["lle", "mds", "pca", "spectral_embedding", "tsne"].')
    args = argparser.parse_args()

    p = Config(args.config)

    outdir = p.outdir + '/projections'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

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

    embeddings = [x for x in os.listdir(p.outdir) if x.startswith('embeddings') and x.endswith('.csv')]

    for e in embeddings:
        segtype, l = emb_file.match(e).groups()
        outdir_cur = outdir + '/' + segtype + '/l' + l + '/' + args.method
        if not os.path.exists(outdir_cur):
            os.makedirs(outdir_cur)
        df = pd.read_csv(p.outdir + '/' + e, sep=' ')

        sys.stderr.write('Projecting using %s. Segtype = %s. Layer = %s. Segments = %d.\n' % (args.method.upper(), segtype, l, len(df)))
        sys.stderr.flush()

        df = project_matching_segments(df, method=args.method)

        sys.stderr.write('Plotting...\n')
        sys.stderr.flush()

        plot_projections(
            df,
            label_map=label_map,
            feature_table=feature_table,
            feature_names=feature_names,
            directory=outdir_cur,
            prefix='l%s_' % l,
            suffix='.png'
        )


