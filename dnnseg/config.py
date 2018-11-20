import sys
import os
import shutil
import numpy as np
import configparser

from .kwargs import UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS



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
        self.order = data.getint('order', 2)
        self.save_preprocessed_data = data.getboolean('save_preprocessed_data', True)

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

        # Process config settings
        self.model_settings = self.build_unsupervised_word_classifier_settings(settings)
        self.model_settings['n_iter'] = settings.getint('n_iter', 1000)
        gpu_frac = settings.get('gpu_frac', None)
        if gpu_frac in [None, 'None']:
            gpu_frac = None
        else:
            try:
                gpu_frac = float(gpu_frac)
            except:
                raise ValueError('gpu_frac parameter invalid: %s' % gpu_frac)
        self.model_settings['gpu_frac'] = gpu_frac
        self.model_settings['use_gpu_if_available'] = settings.getboolean('use_gpu_if_available', True)


    def __getitem__(self, item):
            return self.model_settings[item]

    def build_unsupervised_word_classifier_settings(self, settings):
        out = {}

        # Core fields
        out['network_type'] = settings.get('network_type', 'bayes')

        # Parent class initialization keyword arguments
        out['k'] = settings.getint('k', 128)
        out['outdir'] = self.outdir
        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
            out[kwarg.key] = kwarg.kwarg_from_config(settings)

        # MLE initialization keyword arguments
        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS:
            out[kwarg.key] = kwarg.kwarg_from_config(settings)

        # Bayes initialization keyword arguments
        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS:
            out[kwarg.key] = kwarg.kwarg_from_config(settings)

        return out


