import pandas as pd
import sys
import os
import argparse

from dnnseg.data import segment_table_to_csv

def check_path(path):
    if os.path.basename(path).startswith('embeddings_pred') and path.endswith('.csv'):
        return True
    return False

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Convert CSV segment tables into a format acceptable for the Zerospeech 2015 TDE eval.
    ''')
    argparser.add_argument('paths', nargs='+', help='Paths to CSV files or directories to recursively search for CSV segment table files.')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Write progress report to standard error.')
    args = argparser.parse_args()

    csvs = set()

    for path in args.paths:
        if check_path(path):
            csvs.add(path)
        else:
            for root, _, files in os.walk(path):
                for f in files:
                    p = os.path.join(root, f)
                    if check_path(p):
                        csvs.add(p)

    csvs = sorted(list(csvs))

    for csv in csvs:
        if args.verbose:
            sys.stderr.write('Converting file %s...\n' % csv)
        df = pd.read_csv(csv, sep=' ')
        out = segment_table_to_csv(df, verbose=args.verbose)

        with open(csv[:-4] + '.classes', 'w') as f:
            f.write(out)




