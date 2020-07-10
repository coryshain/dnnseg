import os
import argparse


pbs_base = """#PBS -l walltime=%d:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=%dGB

module load %s
source activate %s
cd %s
"""


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generates PBS job scripts from a set of paths to *.ini files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to *.ini file(s).')
    argparser.add_argument('-t', '--walltime', default='6', type=int, help='Max number of hours for the job.')
    argparser.add_argument('-m', '--memory', default='16', type=int, help='Memory (number of GB) for the job.')
    argparser.add_argument('-l', '--lang', default='eng', type=str, help='Training language.')
    argparser.add_argument('-P', '--python_module', default='python/3.7-conda4.5', help='Name of Python module to load in package manager.')
    argparser.add_argument('-C', '--conda_env', default='dnnseg', help='Name of conda environment to activate.')

    args = argparser.parse_args()

    for path in args.paths:
        basename = os.path.basename(path[:-4])
        pbs_path = path[:-4] + '.pbs'

        with open(pbs_path, 'w') as f:
            f.write('#PBS -N %s\n' % basename)
            f.write(pbs_base % (args.walltime, args.memory, args.python_module, args.conda_env, os.getcwd()))
            f.write('python3 -m dnnseg.bin.train %s\n' % path)
            if args.lang.lower().startswith('eng'):
                f.write('python3 -m dnnseg.bin.evaluate %s -b -d ../dnnseg_data/zerospeech/english -N eng -v\n' % path)
                f.write('python3 -m dnnseg.bin.evaluate %s -b -d ../dnnseg_data/zerospeech/xitsonga -N xit -v\n' % path)
            else:
                f.write('python3 -m dnnseg.bin.evaluate %s -b -d ../dnnseg_data/zerospeech/xitsonga -N xit -v\n' % path)
                f.write('python3 -m dnnseg.bin.evaluate %s -b -d ../dnnseg_data/zerospeech/english -N eng -v\n' % path)
            # f.write('python3 -m dnnseg.bin.plot %s\n' % path)
