import sys
import os
import itertools
import argparse


def compute_hypercube(windows, sizes, discretizations):
    params = []

    if len(windows) > 0:
        params.append([])
        for w in windows:
            name = 'w%s' % w
            w = w.split('-')
            if len(w) == 1:
                w = [w, w]
            bwd, fwd = w
            params[-1].append([('lm_order_bwd', bwd), ('lm_order_fwd', fwd), name])

    if len(sizes) > 0:
        params.append([])
        for s in sizes:
            name = 's%s' % s
            s = s.split('-')
            k = s[-1]
            s = s[:-1]
            n = len(s)
            params[-1].append([('n_layers_encoder', n), ('n_units_encoder', ' '.join(s)), ('k', k), name])

    if len(discretizations) > 0:
        params.append([])
        for d in discretizations:
            name = 'd%s' % d
            if d.lower() == 'sample':
                d = 'True'
            elif d.lower() == 'round':
                d = 'False'
            else:
                raise ValueError('Unrecognized discretization value "%d".' % d)

            params[-1].append([('sample_at_train', d), name])

    out = itertools.product(*params)

    for x in out:
        new = []
        name = []
        for y in x:
            new += y[:-1]
            name.append(y[-1])
        new = dict(new)
        name = '_'.join(name)

        yield new, name


pbs_base = """#PBS -l walltime=%d:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=%dGB

module load %s
source activate %s
cd %s
"""


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generates model (*.ini) files from a source template, using Cartesian project of parameter settings.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to template *.ini file.')
    argparser.add_argument('-w', '--window', default=[], nargs='*', help='Reconstruction window (int or "-"-delimited pair, fwd vs. bwd).')
    argparser.add_argument('-s', '--layer_size', default=[], nargs='*', help='Size of encoder layers (int or "-"-delimited tuple). Also determines number of layers.')
    argparser.add_argument('-d', '--discretization_type', default=[], nargs='*', help='Type of discretization (one of ["sample", "round"]).')
    argparser.add_argument('-t', '--walltime', default='24', type=int, help='Max number of hours for the job.')
    argparser.add_argument('-m', '--memory', default='16', type=int, help='Memory (number of GB) for the job.')
    argparser.add_argument('-P', '--python_module', default='python/3.7-conda4.5', help='Name of Python module to load in package manager.')
    argparser.add_argument('-C', '--conda_env', default='ml', help='Name of conda environment to activate.')
    argparser.add_argument('-O', '--outdir', default='../results/dnnseg', help='Path to output directory.')

    args = argparser.parse_args()

    for path in args.paths:
        params = compute_hypercube(args.window, args.layer_size, args.discretization_type)
        with open(path, 'r') as f:
            lines = f.readlines()
            
        for p, n in params:
            basename = os.path.basename(path[:-4] + '_' + n)
            ini_path = path[:-4] + '_' + n + '.ini'
            pbs_path = path[:-4] + '_' + n + '.pbs'
            outdir = args.outdir
            if not outdir[-1] == '/':
                outdir += '/'
            results_path = outdir + os.path.basename(path[:-4]) + '/' + n
            p['outdir'] = results_path
            with open(ini_path, 'w') as f:
                for l in lines:
                    K= list(p.keys())
                    replace = False
                    for k in K:
                        if l.split('=')[0].strip().lower() == k:
                            f.write(k + ' = ' + p[k] + '\n')
                            del p[k]
                            replace = True
                    if not replace:
                        f.write(l)
                for k in p:
                    f.write(k + ' = ' + p[k] + '\n')

            with open(pbs_path, 'w') as f:
                f.write('#PBS -N %s\n' % basename)
                f.write(pbs_base % (args.walltime, args.memory, args.python_module, args.conda_env, os.getcwd()))
                f.write('python3 -m dnnseg.bin.train %s\n' % ini_path)



