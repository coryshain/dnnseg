import sys
import os
import shutil
import pandas as pd
import argparse

argparser = argparse.ArgumentParser('''
Compiles Zerospeech data for processing with DNN-Seg.
''')
argparser.add_argument('zs', help='Path to Zerospeech 2015 challenge metadata repository (https://github.com/bootphon/Zerospeech2015)')
argparser.add_argument('tde', help='Path to Zerospeech 2015 challenge Track 2 repository (https://github.com/bootphon/tde)')
argparser.add_argument('-b', '--bsc', type=str, default=None, help='Path to Buckeye Speech Corpus')
argparser.add_argument('-x', '--xit', type=str, default=None, help='Path to directory containing Xitsonga portion of NCHLT')
argparser.add_argument('-o', '--outdir', default='./', help='Path to output directory (if not specified, uses current working directory)')
args = argparser.parse_args()

sample_files = []
english_files = []
xitsonga_files = []

with open(args.zs + '/sample_files.txt', 'r') as f:
    for l in f.readlines():
        filename = l.strip()[:-4]
        sample_files.append(filename)

with open(args.zs + '/english_files.txt', 'r') as f:
    for l in f.readlines():
        filename = l.strip()[:-4]
        if filename != 's0103a':
            english_files.append(filename)

with open(args.zs + '/xitsonga_files.txt', 'r') as f:
    for l in f.readlines():
        xitsonga_files.append(l.strip()[:-4])

sample_vad = pd.read_csv(args.zs + '/sample_vad.txt', sep=' ', header=None, names=['fileID', 'start', 'end'])
sample_vad.start = sample_vad.start.round(3)
sample_vad.end = sample_vad.end.round(3)
sample_wrd = pd.read_csv(args.tde + 'bin/resources/sample.wrd', sep=' ', header=None, names=['fileID', 'start', 'end', 'label'])
sample_wrd.start = sample_wrd.start.round(3)
sample_wrd.end = sample_wrd.end.round(3)
sample_phn = pd.read_csv(args.tde + 'bin/resources/sample.phn', sep=' ',  header=None, names=['fileID', 'start', 'end', 'label'])
sample_phn.start = sample_phn.start.round(3)
sample_phn.end = sample_phn.end.round(3)

english_vad = pd.read_csv(args.zs + '/english_vad.txt', sep=',', header=0, names=['fileID', 'start', 'end'])
english_vad.start = english_vad.start.round(3)
english_vad.end = english_vad.end.round(3)
english_wrd = pd.read_csv(args.tde + 'bin/resources/english.wrd', sep=' ', header=None, names=['fileID', 'start', 'end', 'label'])
english_wrd.start = english_wrd.start.round(3)
english_wrd.end = english_wrd.end.round(3)
english_phn = pd.read_csv(args.tde + 'bin/resources/english.phn', sep=' ', header=None, names=['fileID', 'start', 'end', 'label'])
english_phn.start = english_phn.start.round(3)
english_phn.end = english_phn.end.round(3)

xitsonga_vad = pd.read_csv(args.zs + '/xitsonga_vad.txt', sep=' ', header=None, names=['fileID', 'start', 'end'])
xitsonga_vad.start = xitsonga_vad.start.round(3)
xitsonga_vad.end = xitsonga_vad.end.round(3)
xitsonga_wrd = pd.read_csv(args.tde + 'bin/resources/xitsonga.wrd', sep=' ', header=None, names=['fileID', 'start', 'end', 'label'])
xitsonga_wrd.start = xitsonga_wrd.start.round(3)
xitsonga_wrd.end = xitsonga_wrd.end.round(3)
xitsonga_phn = pd.read_csv(args.tde + 'bin/resources/xitsonga.phn', sep=' ', header=None, names=['fileID', 'start', 'end', 'label'])
xitsonga_phn.start = xitsonga_phn.start.round(3)
xitsonga_phn.end = xitsonga_phn.end.round(3)


sys.stderr.write('Processing Buckeye Speech Corpus data...\n')
if args.bsc is not None:
    if not os.path.exists(args.outdir + '/zerospeech/sample'):
        os.makedirs(args.outdir + '/zerospeech/sample')

    for fileID in sample_files:
        subject = fileID[:3]
        in_path = args.bsc + '/' + subject + '/' + fileID + '/' + fileID + '.wav'
        out_path = args.outdir + '/zerospeech/sample/' + fileID + '.wav'
        shutil.copy2(in_path, out_path)

        to_print = sample_vad[sample_vad.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/sample/%s.vad' %fileID, sep=' ', index=False)

        to_print = sample_wrd[sample_wrd.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/sample/%s.wrd' %fileID, sep=' ', index=False)

        to_print = sample_phn[sample_phn.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/sample/%s.phn' %fileID, sep=' ', index=False)

    if not os.path.exists(args.outdir + '/zerospeech/english'):
        os.makedirs(args.outdir + '/zerospeech/english')

    for fileID in english_files:
        subject = fileID[:3]
        in_path = args.bsc + '/' + subject + '/' + fileID + '/' + fileID + '.wav'
        out_path = args.outdir + '/zerospeech/english/' + fileID + '.wav'
        shutil.copy2(in_path, out_path)

        to_print = english_vad[english_vad.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/english/%s.vad' % fileID, sep=' ', index=False)

        to_print = english_wrd[english_wrd.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/english/%s.wrd' % fileID, sep=' ', index=False)

        to_print = english_phn[english_phn.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/english/%s.phn' % fileID, sep=' ', index=False)

else:
    sys.stderr.write('No path provided to Buckeye Speech Corpus. Skipping...\n')

sys.stderr.write('Processing NCHST (Xitsonga) data...\n')
if args.xit is not None:
    if not os.path.exists(args.outdir + '/zerospeech/xitsonga'):
        os.makedirs(args.outdir + '/zerospeech/xitsonga')

    for fileID in xitsonga_files:
        subject = fileID[10:13]
        in_path = args.xit + '/audio/' + subject + '/' + fileID + '.wav'
        out_path = args.outdir + '/zerospeech/xitsonga/' + fileID + '.wav'
        shutil.copy2(in_path, out_path)

        to_print = xitsonga_vad[xitsonga_vad.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/xitsonga/%s.vad' % fileID, sep=' ', index=False)

        to_print = xitsonga_wrd[xitsonga_wrd.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/xitsonga/%s.wrd' % fileID, sep=' ', index=False)

        to_print = xitsonga_phn[xitsonga_phn.fileID == fileID]
        to_print['speaker'] = subject
        to_print.to_csv(args.outdir + '/zerospeech/xitsonga/%s.phn' % fileID, sep=' ', index=False)

else:
    sys.stderr.write('No path provided to Xitsonga data. Skipping...\n')
