import os
import yaml
import itertools
import argparse


def compute_hypercube(yml):
    params = {}
    for name in yml:
        v = yml[name]
        if isinstance(v, list):
            out_cur = {name: v}
        elif isinstance(v, dict):
            out_cur = v
        else:
            raise ValueError('Value of unrecognized type %d. Must be ``list`` or ``dict``. Value:\n%s' %(type(v), v))

        params[''.join(name.split('_'))] = out_cur

    names = sorted(list(params.keys()))
    search = []
    for n in names:
        search_cur = []
        if 'names' in params[n]:
            val_names = params[n].pop('names')
        else:
            val_names = []
        for k in params[n]:
            for i in range(len(params[n][k])):
                v = str(params[n][k][i])
                if i >= len(val_names):
                    val_name = v
                    val_names.append(val_names)
                else:
                    val_name = val_names[i]
                if i >= len(search_cur):
                    search_cur.append([n, val_name, (k, v)])
                else:
                    search_cur[i].append((k, v))
        search.append(search_cur)

    out = itertools.product(*search)

    for x in out:
        new = []
        name = []
        for y in x:
            new += y[2:]
            name.append(''.join(y[:2]))
        new = dict(new)
        name = '_'.join(name)

        yield new, name


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generates model (*.ini) files from source templates, using Cartesian project of parameter settings defined in a
        YAML file. The YAML file must define a dictionary in which keys contain the user-specified name of the search
        dimension and values contain dictionaries from DNNSEG hyperparameter keys to lists of values to search.
        Multiple dictionaries given under the same name will be interpreted as co-varying, rather than being searched
        in a grid. An optional reserved key ``names`` allows the user to specify the name of each level in the
        search over a dimension. If omitted, the values of the levels from one of the dictionaries will be chosen.
        As a shortcut, an entry in the YAML file can also consist of a single map from a DNNSEG hyperparameter
        key to a list of values, in which case the key will also be used as the name for the search dimension.
        
        For example, to search over encoder depths vs. covarying levels of state and boundary noise, the following
        YAML string would be used:
        
        n_layers_encoder:
          - 2
          - 3
          - 4
        noise:
          names:
            - L
            - M
            - H
          encoder_state_noise_sd:
            - 0.
            - 0.25
            - 0.5
          encoder_boundary_noise_sd:
            - 0.
            - 0.1
            - 0.25
        
        This will create a 3 x 3 search over number of encoder layers vs. levels of noise, with noise levels treated as 
        covarying -- i.e. L = (0,0), M = (0.25, 0.1), and H = (0.5, 0.25) for state and boundary noise levels, respectively.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to template *.ini file(s).')
    argparser.add_argument('search_params_path', help='Path to YAML file defining search params.')
    argparser.add_argument('-o', '--ini_outdir', default='./', help='Path to directory in which to put generated *.ini files.')
    argparser.add_argument('-O', '--results_outdir', default='../results/dnnseg', help='Path to directory in which to save models.')

    args = argparser.parse_args()
    
    os.makedirs(args.ini_outdir)
    os.makedirs(args.results_outdir)

    with open(args.search_params_path, 'r') as f:
        yml = yaml.load(f)

    for path in args.paths:
        params = compute_hypercube(yml)
        with open(path, 'r') as f:
            lines = f.readlines()
            
        for p, n in params:
            ini_outdir = args.ini_outdir
            if ini_outdir[-1] != '/':
                ini_outdir += '/'
            basename = os.path.basename(path[:-4] + '_' + n)
            ini_path = ini_outdir + basename + '.ini'
            outdir = args.results_outdir
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
