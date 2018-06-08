import sys
import os
import shutil
import numpy as np
import configparser


class Config(object):
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)

        # Data
        data = config['data']
        self.train_data_dir = data.get('train_data_dir', './')
        self.dev_data_dir = data.get('dev_data_dir', './')
        self.test_data_dir = data.get('test_data_dir', './')
        self.n_coef = data.getint('n_coef', 13)
        self.order = data.get('order', None)
        if self.order is None or self.order == 'None':
            self.order = [1, 2]
        else:
            self.order = sorted(list(set([int(x) for x in self.order.split()])))

        # SETTINGS
        # Output directory
        settings = config['settings']
        self.outdir = settings.get('outdir', None)
        if self.outdir is None:
            self.outdir = settings.get('logdir', None)
        if self.outdir is None:
            self.outdir = './dtsr_model/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if os.path.realpath(path) != os.path.realpath(self.outdir + '/config.ini'):
            shutil.copy2(path, self.outdir + '/config.ini')

        # Model hyperparameters
        self.k = settings.getint('k', 128)
        self.temp = settings.getfloat('temp', 1.)
        self.trainable_temp = settings.getboolean('trainable_temp', False)
        self.binary_classifier = settings.getboolean('binary_classifier', False)
        self.emb_dim = settings.get('emb_dim', None)
        if self.emb_dim in [None, 'None']:
            self.emb_dim = None
        else:
            try:
                self.emb_dim = int(self.emb_dim)
            except:
                raise ValueError('emb_dim parameter invalid: %s' %self.emb_dim)
        self.encoder_type = settings.get('encoder_type', 'rnn')
        self.decoder_type = settings.get('decoder_type', 'rnn')
        self.dense_n_layers = settings.getint('dense_n_layers', 1)
        self.conv_n_filters = settings.getint('conv_n_filters', 16)
        self.conv_kernel_size = settings.getint('conv_kernel_size', 3)
        self.output_scale = settings.get('output_scale', None)
        if self.output_scale in [None, 'None']:
            self.output_scale = None
        else:
            try:
                self.output_scale = float(self.output_scale)
            except:
                raise ValueError('output_scale parameter invalid: %s' %self.output_scale)
        self.unroll_rnn = settings.getboolean('unroll_rnn', False)
        self.reconstruct_deltas = settings.getboolean('reconstruct_deltas', False)

        # Optimizer
        self.optim = settings.get('optim', 'Adam')
        if self.optim == 'None':
            self.optim = None
        self.learning_rate = settings.getfloat('learning_rate', 0.001)
        self.learning_rate_min = settings.get('learning_rate_min', 1e-5)
        if self.learning_rate_min in [None, 'None', '-inf']:
            self.learning_rate_min = -np.inf
        else:
            try:
                self.learning_rate_min = float(self.learning_rate_min)
            except:
                raise ValueError('learning_rate_min parameter invalid: %s' % self.learning_rate_min)
        self.lr_decay_family = settings.get('lr_decay_family', None)
        if self.lr_decay_family == 'None':
            self.lr_decay_family = None
        self.lr_decay_steps = settings.getint('lr_decay_steps', 100)
        self.lr_decay_rate = settings.getfloat('lr_decay_rate', .5)
        self.lr_decay_staircase = settings.getboolean('lr_decay_staircase', False)
        self.max_global_gradient_norm = settings.get('max_global_gradient_norm', None)
        if self.max_global_gradient_norm in [None, 'None', 'inf']:
            self.max_global_gradient_norm = None
        else:
            try:
                self.max_global_gradient_norm = float(self.max_global_gradient_norm)
            except:
                raise ValueError('max_global_gradient_norm parameter invalid: %s' % self.max_global_gradient_norm)
        self.init_sd = settings.getfloat('init_sd', 1.)
        self.ema_decay = settings.getfloat('ema_decay', 0.999)

        # Implementation
        self.gpu_frac = settings.get('gpu_frac', None)
        if self.gpu_frac is not None and self.gpu_frac != 'None':
            self.gpu_frac = float(self.gpu_frac)
        self.use_gpu_if_available = settings.getboolean('use_gpu_if_available', True)
        self.float_type = settings.get('float_type', 'float32')
        self.int_type = settings.get('int_type', 'int32')
        self.save_preprocessed_data = settings.getboolean('save_preprocessed_data', True)
        self.n_iter = settings.getint('n_iter', 1000)
        self.n_samples = settings.getint('n_samples', 2)
        self.minibatch_size = settings.getint('minibatch_size', 128)
        self.eval_minibatch_size = settings.getint('eval_minibatch_size', 100000)
        self.log_freq = settings.getint('log_freq', 1)
        self.save_freq = settings.getint('save_freq', 1)






    def __str__(self):
        out = ''
        V = vars(self)
        for x in V:
            out += '%s: %s\n' %(x, V[x])
        return out

