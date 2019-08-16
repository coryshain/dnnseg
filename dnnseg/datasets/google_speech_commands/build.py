import sys
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import argparse

argparser = argparse.ArgumentParser('''
Builds Google Speech Commands dataset into appropriate format for use with DNN-Seg.
''')
argparser.add_argument('dir_path', help='Path to Google Speech Commands source directory')
argparser.add_argument('-o', '--outdir', default='../dnnseg_data/google_speech_commands/', help='')
args = argparser.parse_args()

if args.outdir is None:
    args.outdir = os.path.dirname(args.dir_path)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
if not os.path.exists(args.outdir + '/train'):
    os.makedirs(args.outdir + '/train')
if not os.path.exists(args.outdir + '/dev'):
    os.makedirs(args.outdir + '/dev')
if not os.path.exists(args.outdir + '/test'):
    os.makedirs(args.outdir + '/test')

commands = [x for x in os.listdir(args.dir_path) if not x.startswith('_') and os.path.isdir(args.dir_path + '/' + x)]
SR = 16000
WINDOW_LEN = 0.1
PAD_LEN = int(SR * WINDOW_LEN)

pad = np.zeros(PAD_LEN, dtype=np.int16)

dev_files = {}
with open(args.dir_path + '/validation_list.txt', 'r') as f:
    for line in f.readlines():
        command, file = line.strip().split('/')
        if command not in dev_files:
            dev_files[command] = [file]
        else:
            dev_files[command].append(file)

test_files = {}
with open(args.dir_path + '/testing_list.txt', 'r') as f:
    for line in f.readlines():
        command, file = line.strip().split('/')
        if command not in test_files:
            test_files[command] = [file]
        else:
            test_files[command].append(file)

for c in commands:
    sys.stderr.write('Processing subdirectory "%s"...\n' %c)
    files = os.listdir(args.dir_path + '/' + c + '/')
    n = len(files)
    dev_list = dev_files[c]
    test_list = test_files[c]

    train_wavs = []
    train_start = []
    train_end = []
    train_label = []
    train_speaker = []

    dev_wavs = []
    dev_start = []
    dev_end = []
    dev_label = []
    dev_speaker = []

    test_wavs = []
    test_start = []
    test_end = []
    test_label = []
    test_speaker = []

    for i, f in enumerate(files):
        ID = c + '_' + os.path.basename(f)[:-4]
        sys.stderr.write('\r%d/%d' %(i+1, n))
        sys.stderr.flush()
        speaker, _, _ = f.split('_')
        _, wav = wavfile.read(args.dir_path + '/' + c + '/' + f)

        if f in dev_list:
            partition = 'dev'
            if len(dev_wavs) > 0:
                dev_wavs.append(pad)
                dev_wavs.append(wav)
                dev_start.append(dev_end[-1] + float(PAD_LEN) / SR)
                dev_end.append(dev_start[-1] + float(len(wav)) / SR)
            else:
                dev_wavs.append(wav)
                dev_start.append(0.)
                dev_end.append(float(len(wav)) / SR)
            dev_label.append(c)
            dev_speaker.append(speaker)
        elif f in test_list:
            partition = 'test'
            if len(test_wavs) > 0:
                test_wavs.append(pad)
                test_wavs.append(wav)
                test_start.append(test_end[-1] + float(PAD_LEN) / SR)
                test_end.append(test_start[-1] + float(len(wav)) / SR)
            else:
                test_wavs.append(wav)
                test_start.append(0.)
                test_end.append(float(len(wav)) / SR)
            test_label.append(c)
            test_speaker.append(speaker)
        else:
            partition = 'train'
            if len(train_wavs) > 0:
                train_wavs.append(pad)
                train_wavs.append(wav)
                train_start.append(train_end[-1] + float(PAD_LEN) / SR)
                train_end.append(train_start[-1] + float(len(wav)) / SR)
            else:
                train_wavs.append(wav)
                train_start.append(0.)
                train_end.append(float(len(wav)) / SR)
            train_label.append(c)
            train_speaker.append(speaker)

    sys.stderr.write('\nSaving preprocessed data...\n')

    train_data = pd.DataFrame(
        {
            'start': train_start,
            'end': train_end,
            'label': train_label,
            'speaker': train_speaker,
            'index': np.arange(len(train_start))
        }
    )
    train_data.start = train_data.start.round(decimals=3)
    train_data.end = train_data.end.round(decimals=3)
    train_data.to_csv(args.outdir + '/train/%s.wrd' %c, sep=' ', index=False, columns=['start', 'end', 'label', 'speaker', 'index'])
    train_wavs = np.concatenate(train_wavs, axis=0)
    wavfile.write(args.outdir + '/train/%s.wav' % c, SR, train_wavs)

    dev_data = pd.DataFrame(
        {
            'start': dev_start,
            'end': dev_end,
            'label': dev_label,
            'speaker': dev_speaker,
            'index': np.arange(len(dev_start))
        }
    )
    dev_data.start = dev_data.start.round(decimals=3)
    dev_data.end = dev_data.end.round(decimals=3)
    dev_data.to_csv(args.outdir + '/dev/%s.wrd' %c, sep=' ', index=False, columns=['start', 'end', 'label', 'speaker', 'index'])
    dev_wavs = np.concatenate(dev_wavs, axis=0)
    wavfile.write(args.outdir + '/dev/%s.wav' % c, SR, dev_wavs)

    test_data = pd.DataFrame(
        {
            'start': test_start,
            'end': test_end,
            'label': test_label,
            'speaker': test_speaker,
            'index': np.arange(len(test_start))
        }
    )
    test_data.start = test_data.start.round(decimals=3)
    test_data.end = test_data.end.round(decimals=3)
    test_data.to_csv(args.outdir + '/test/%s.wrd' %c, sep=' ', index=False, columns=['start', 'end', 'label', 'speaker', 'index'])
    test_wavs = np.concatenate(test_wavs, axis=0)
    wavfile.write(args.outdir + '/test/%s.wav' % c, SR, test_wavs)

