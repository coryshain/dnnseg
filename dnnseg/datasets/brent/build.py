import sys
import os
import argparse

argparser = argparse.ArgumentParser('''
Builds Brent dataset into appropriate format for use with DNN-Seg.
''')
argparser.add_argument('dir_path', help='Path to Brent source directory')
argparser.add_argument('-o', '--outdir', default='../dnnseg_data/brent/', help='')
args = argparser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
if not os.path.exists(args.outdir + '/train'):
    os.makedirs(args.outdir + '/train')
if not os.path.exists(args.outdir + '/test'):
    os.makedirs(args.outdir + '/test')

with open(args.dir_path + '/br-phono.txt', 'r') as f:
    with open(args.outdir + '/train/br-phono.txt', 'w') as train:
        with open(args.outdir + '/test/br-phono.txt', 'w') as test:
            for i, l in enumerate(f.readlines()):
                if l.strip():
                    if i < 8000:
                        train.write(l)
                    else:
                        test.write(l)