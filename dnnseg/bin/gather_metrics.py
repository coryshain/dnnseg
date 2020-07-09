import re
import os
import yaml
import pandas as pd
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Collect evaluation metrics from multiple models and dump them into a table.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to directory/directories to search for evaluation reports.')
    argparser.add_argument('-m', '--match', type=str, help='Regex that the report paths must match.')
    argparser.add_argument('-o', '--outpath', default='./eval_table.csv', help='Path to output file.')
    args = argparser.parse_args()

    out = []

    if args.match:
        print(args.match)
        match = re.compile(args.match)


    for path in args.paths:
        for root, _, files in os.walk(path):
            for x in files:
                path_str = os.path.join(root, x)
                if x.startswith('eval_table') and x.endswith('yml') and (not args.match or match.match(path_str)):
                    with open(path_str, 'r') as f:
                        out.append(yaml.load(f))

    out = pd.DataFrame(out)
    out.to_csv(args.outpath, sep=' ', index=False, na_rep='NaN')



