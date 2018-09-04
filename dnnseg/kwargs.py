from functools import cmp_to_key

class Kwarg(object):
    """
    Data structure for storing keyword arguments and their docstrings.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param dtypes: ``list`` or ``class``; List of classes or single class. Members can also be specific required values, either ``None`` or values of type ``str``.
    :param descr: ``str``; Description of kwarg
    """

    def __init__(self, key, default_value, dtypes, descr, aliases=None):
        if aliases is None:
            aliases = []
        self.key = key
        self.default_value = default_value
        if not isinstance(dtypes, list):
            self.dtypes = [dtypes]
        else:
            self.dtypes = dtypes
        self.dtypes = sorted(self.dtypes, key=cmp_to_key(Kwarg.type_comparator))
        self.descr = descr
        self.aliases = aliases

    def dtypes_str(self):
        if len(self.dtypes) == 1:
            out = '``%s``' %self.get_type_name(self.dtypes[0])
        elif len(self.dtypes) == 2:
            out = '``%s`` or ``%s``' %(self.get_type_name(self.dtypes[0]), self.get_type_name(self.dtypes[1]))
        else:
            out = ', '.join(['``%s``' %self.get_type_name(x) for x in self.dtypes[:-1]]) + ' or ``%s``' %self.get_type_name(self.dtypes[-1])

        return out

    def get_type_name(self, x):
        if isinstance(x, type):
            return x.__name__
        if isinstance(x, str):
            return '"%s"' %x
        return str(x)

    def in_settings(self, settings):
        out = False
        if self.key in settings:
            out = True

        if not out:
            for alias in self.aliases:
                if alias in settings:
                    out = True
                    break

        return out

    def kwarg_from_config(self, settings):
        if len(self.dtypes) == 1:
            val = {
                str: settings.get,
                int: settings.getint,
                float: settings.getfloat,
                bool: settings.getboolean
            }[self.dtypes[0]](self.key, None)

            if val is None:
                for alias in self.aliases:
                    val = {
                        str: settings.get,
                        int: settings.getint,
                        float: settings.getfloat,
                        bool: settings.getboolean
                    }[self.dtypes[0]](alias, self.default_value)
                    if val is not None:
                        break

            if val is None:
                val = self.default_value

        else:
            from_settings = settings.get(self.key, None)
            if from_settings is None:
                for alias in self.aliases:
                    from_settings = settings.get(alias, None)
                    if from_settings is not None:
                        break

            if from_settings is None:
                val = self.default_value
            else:
                parsed = False
                for x in reversed(self.dtypes):
                    if x == None:
                        if from_settings == 'None':
                            val = None
                            parsed = True
                            break
                    elif isinstance(x, str):
                        if from_settings == x:
                            val = from_settings
                            parsed = True
                            break
                    else:
                        try:
                            val = x(from_settings)
                            parsed = True
                            break
                        except:
                            pass

                assert parsed, 'Invalid value "%s" received for %s' %(from_settings, self.key)

        return val



    @staticmethod
    def type_comparator(a, b):
        '''
        Types precede strings, which precede ``None``
        :param a: First element
        :param b: Second element
        :return: ``-1``, ``0``, or ``1``, depending on outcome of comparison
        '''
        if isinstance(a, type) and not isinstance(b, type):
            return -1
        elif not isinstance(a, type) and isinstance(b, type):
            return 1
        elif isinstance(a, str) and not isinstance(b, str):
            return -1
        elif isinstance(b, str) and not isinstance(a, str):
            return 1
        else:
            return 0





UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS = [
    Kwarg(
        'outdir',
        './dtsr_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
    ),
    Kwarg(
        'binary_classifier',
        False,
        bool,
        "Implement the classifier as a binary code in which categories can share bits. If ``False``, implements the classifier using independent categories."
    ),
    Kwarg(
        'emb_dim',
        None,
        [int, None],
        "Append **emb_dim** ELU-activated pass-through channels to the classifier output for capturing category-internal variation. If ``None`` or ``0``, the encoding consists exclusively of the classifier output."
    ),
    Kwarg(
        'utt_len_emb_dim',
        None,
        [int, None],
        "Append **utt_len_emb_dim** embedding of utterance length (number of non-padding characters) to the classifier output for capturing temporal dilation. If ``None`` or ``0``, the encoding consists exclusively of the classifier output."
    ),
    Kwarg(
        'encoder_type',
        'rnn',
        str,
        "Encoder network to use. One of ``dense``, ``cnn``, or ``rnn``."
    ),
    Kwarg(
        'decoder_type',
        'rnn',
        str,
        "Decoder network to use. One of ``dense``, ``cnn``, or ``rnn``."
    ),
    Kwarg(
        'n_layers_encoder',
        2,
        int,
        "Number of layers to use for encoder. Ignored if **encoder_type** is not ``dense``."
    ),
    Kwarg(
        'n_layers_decoder',
        2,
        int,
        "Number of layers to use for decoder. Ignored if **decoder_type** is not ``dense``."
    ),
    Kwarg(
        'unroll',
        True,
        bool,
        "Unroll any RNN layers."
    ),
    Kwarg(
        'conv_n_filters',
        16,
        int,
        "Number of filters to use in convolutional layers. Ignored if no residual layers in the model."
    ),
    Kwarg(
        'conv_kernel_size',
        3,
        int,
        "Size of kernel to use in convolutional layers. Ignored if no residual layers in the model."
    ),
    Kwarg(
        'resnet_n_layers_inner',
        2,
        int,
        "Number of internal layers to use in residual layers. Ignored if no residual layers in the model."
    ),
    Kwarg(
        'batch_normalize',
        True,
        bool,
        "Apply batch normalization where appropriate."
    ),
    Kwarg(
        'n_coef',
        13,
        int,
        "Number of cepstral coefficients present in the input data"
    ),
    Kwarg(
        'order',
        2,
        int,
        "Maximum order of derivatives present in the input data. All lower orders must also be included."
    ),
    Kwarg(
        'resample',
        None,
        [int, None],
        "Resample inputs and targets to fixed length **resample** timesteps. If ``None``, no resampling."
    ),
    Kwarg(
        'reconstruct_deltas',
        False,
        bool,
        "Include derivatives in the reconstruction targets."
    ),
    Kwarg(
        'normalize_data',
        True,
        bool,
        "Normalize utterances to the range :math:`[0, 1]`"
    ),
    Kwarg(
        'constrain_output',
        True,
        bool,
        "Use an output model constrained to :math:`[0, 1]` (sigmoid if MLE and LogitNormal if Bayesian). Otherwise, use linear/normal output. Requires **normalize_data* to be ``True``."
    ),
    Kwarg(
        'n_timesteps',
        None,
        [int, None],
        "Number of timesteps present in the input data. If ``None``, inferred from data, possibly with different values between batches."
    ),
    Kwarg(
        'mask_padding',
        True,
        bool,
        "Mask padding frames in reconstruction targets so that they are ignored in gradient updates."
    ),
    Kwarg(
        'decoder_use_input_means',
        False,
        bool,
        "In addition to classifier's encoding, provide mean activations across the spectral and time dimensions to the decoder."
    ),
    Kwarg(
        'decoder_use_input_length',
        False,
        bool,
        "Explicitly pass number of non-padding frames in input as feature to the decoder."
    ),
    Kwarg(
        'residual_decoder',
        False,
        bool,
        "Compute the decoding as a sum of the network outputs and the mean activations of all cells in the training data."
    ),
    Kwarg(
        'per_class_decoder',
        False,
        bool,
        "Use separate decoders for each class and merge their outputs as a sum weighted by the class probability."
    ),
    Kwarg(
        'optim_name',
        'Nadam',
        [str, None],
        """Name of the optimizer to use. Must be one of:
        
            - ``'SGD'``
            - ``'Momentum'``
            - ``'AdaGrad'``
            - ``'AdaDelta'``
            - ``'Adam'``
            - ``'FTRL'``
            - ``'RMSProp'``
            - ``'Nadam'``
            - ``None`` (DTSRBayes only; uses the default optimizer defined by Edward, which currently includes steep learning rate decay and is therefore not recommended in the general case)"""
    ),
    Kwarg(
        'epsilon',
        1e-5,
        float,
        "Epsilon to avoid boundary violations."
    ),
    Kwarg(
        'optim_epsilon',
        1e-8,
        float,
        "Epsilon parameter to use if **optim_name** in ``['Adam', 'Nadam']``, ignored otherwise."
    ),
    Kwarg(
        'learning_rate',
        0.01,
        float,
        "Initial value for the learning rate."
    ),
    Kwarg(
        'learning_rate_min',
        1e-4,
        float,
        "Minimum value for the learning rate."
    ),
    Kwarg(
        'lr_decay_family',
        None,
        [str, None],
        "Functional family for the learning rate decay schedule (no decay if ``None``)."
    ),
    Kwarg(
        'lr_decay_rate',
        0.,
        float,
        "coefficient by which to decay the learning rate every ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_steps',
        25,
        int,
        "Span of iterations over which to decay the learning rate by ``lr_decay_rate`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_staircase',
        False,
        bool,
        "Keep learning rate flat between ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'max_global_gradient_norm',
        None,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no gradient clipping.'
    ),
    Kwarg(
        'regularizer_name',
        None,
        [str, None],
        "Name of global regularizer. If ``None``, no regularization."
    ),
    Kwarg(
        'regularizer_scale',
        0.01,
        float,
        "Scale of global regularizer (ignored if ``regularizer_name==None``)."
    ),
    Kwarg(
        'ema_decay',
        0.999,
        float,
        "Decay factor to use for exponential moving average for parameters (used in prediction)."
    ),
    Kwarg(
        'minibatch_size',
        128,
        [int, None],
        "Size of minibatches to use for fitting (full-batch if ``None``)."
    ),
    Kwarg(
        'eval_minibatch_size',
        100000,
        [int, None],
        "Size of minibatches to use for prediction/evaluation (full-batch if ``None``)."
    ),
    Kwarg(
        'float_type',
        'float32',
        str,
        "``float`` type to use throughout the network."
    ),
    Kwarg(
        'int_type',
        'int32',
        str,
        "``int`` type to use throughout the network (used for tensor slicing)."
    ),
    Kwarg(
        'save_freq',
        1,
        int,
        "Frequency (in iterations) with which to save model checkpoints."
    ),
    Kwarg(
        'log_graph',
        False,
        bool,
        "Log the network graph to Tensorboard"
    )
]

UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS = []

UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS = [
    Kwarg(
        'inference_name',
        'KLqp',
        str,
        "Name of Edward inference to use. Currently, only variational inferences supported."
    ),
    Kwarg(
        'n_samples',
        2,
        int,
        "Number of samples to draw during inference."
    ),
    Kwarg(
        'n_iter',
        10000,
        int,
        "Expected number of training iterations. Used only for logging purposes."
    ),
    Kwarg(
        'relaxed',
        True,
        bool,
        "Use relaxed implementations of discrete classifier random variables (i.e. ``RelaxedOneHotCategorical`` or ``RelaxedBernoulli``). If ``False``, use discrete random variables for classification."
    ),
    Kwarg(
        'temp',
        1.,
        float,
        "Initial value of classifier's temperature parameter. Used only if **relaxed** is ``True``."
    ),
    Kwarg(
        'trainable_temp',
        True,
        bool,
        "Allow the temperature parameter to be tuned -- otherwise keep fixed at its initialization. Used only if **relaxed** is ``True``."
    ),
    Kwarg(
        'mv',
        False,
        bool,
        "Multivariate output model -- fit covariances between cells in the reconstruction targets."
    ),
    Kwarg(
        'output_scale',
        None,
        [float, None],
        "Scale to use for the output model. If ``None``, a per-category scale model is fitted for each cell in the reconstruction targets. Ignored if **mv** is ``True``."
    ),
    Kwarg(
        'declare_priors',
        True,
        bool,
        "Declare uniform prior over classifier random variable. If ``False``, no explicit priors are declared."
    )
]


def dtsr_kwarg_docstring():
    out = "**Both MLE and Bayes**\n\n"

    for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**MLE only**\n\n'

    for kwarg in UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**Bayes only**\n\n'

    for kwarg in UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n'

    return out
