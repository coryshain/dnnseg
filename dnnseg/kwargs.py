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
        'task',
        'autoencoder',
        str,
        "Task to perform. One of ``['utterance_classifier', 'autoencoder', 'streaming_autoencoder']``."
    ),
    Kwarg(
        'segtype',
        'vad',
        str,
        'Utterance-level segmentation type used to chunk audio input (one of ["vad", "wrd", "phn"]).'
    ),
    Kwarg(
        'n_correspondence',
        None,
        [int, None],
        "Number of discovered segments to use to compute correspondence autoencoder auxiliary loss. If ``0`` or ``None``, do not use correpondence autoencoder."
    ),
    Kwarg(
        'resample_correspondence',
        25,
        int,
        "Number of timesteps to which correspondence autoencoder targets should be resampled. Ignored if **n_correspondence** is ``0`` or ``None``."
    ),
    Kwarg(
        'correspondence_start_iter',
        1,
        int,
        "Iteration number at which to start minimizing correpondence autoencoder auxiliary loss. Ignored if **n_correspondence** is ``0`` or ``None``."
    ),
    Kwarg(
        'correspondence_loss_weight',
        1.,
        float,
        "Coefficient by which to scale correspondence autoencoder auxiliary loss. Ignored if **n_correspondence** is ``0`` or ``None``."
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
        'speaker_emb_dim',
        None,
        [int, None],
        "Append a **speaker_emb_dim** dimensional embedding of the speaker ID to each acoustic frame and to the utterance embedding. If ``None`` or ``0``, no speaker embedding used."
    ),
    Kwarg(
        'speaker_ids',
        '',
        str,
        "Space-delimited list of string IDs for the speakers in the training data. Ignored if **speaker_emb_dim** is ``None`` or ``0``."
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
        'n_layers_encoder',
        2,
        int,
        "Number of layers to use for encoder. Ignored if **encoder_type** is not ``dense``."
    ),
    Kwarg(
        'n_units_encoder',
        None,
        [int, str, None],
        "Number of units to use in non-final encoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_encoder** - 1 space-delimited integers, one for each layer in order from bottom to top, or ``None``, in which case the units will be equal to **k**."
    ),
    Kwarg(
        'encoder_activation',
        'tanh',
        [str, None],
        "Name of activation to use at the output of the encoder",
    ),
    Kwarg(
        'encoder_inner_activation',
        'tanh',
        [str, None],
        "Name of activation to use for any internal layers of the encoder",
        aliases=['inner_activation']
    ),
    Kwarg(
        'encoder_recurrent_activation',
        'hard_sigmoid',
        [str, None],
        "Name of activation to use for recurrent activation in recurrent layers of the encoder. Ignored if encoder is not recurrent.",
        aliases=['recurrent_activation']
    ),
    Kwarg(
        'encoder_boundary_activation',
        'hard_sigmoid',
        [str, None],
        "Name of activation to use for boundary activation in the HM-LSTM encoder. Ignored if encoder is not an HM-LSTM.",
        aliases=['boundary_activation']
    ),
    Kwarg(
        'encoder_weight_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of encoder weight regularization. If ``float``, scale of encoder L2 weight regularization. If ``None``, no encoder weight regularization."
    ),
    Kwarg(
        'boundary_power',
        None,
        [int, None],
        "Power to raise boundary probabilities to for information flow in the HM-LSTM encoder. Ignored if encoder is not an HM-LSTM."
    ),
    Kwarg(
        'boundary_slope_annealing_rate',
        None,
        [float, None],
        "Whether to anneal the slopes of the boundary activations.",
        aliases=['slope_annealing_rate']
    ),
    Kwarg(
        'state_slope_annealing_rate',
        None,
        [float, None],
        "Whether to anneal the slopes of the hidden state activations.",
        aliases=['slope_annealing_rate']
    ),
    Kwarg(
        'slope_annealing_max',
        10.,
        [float, None],
        "Maximum allowed value of the slope annealing coefficient. If ``None``, no maximum will be enforced."
    ),
    Kwarg(
        'encoder_state_discretizer',
        None,
        [str, None],
        "Discretization function to apply to encoder hidden states, one of ``None``, ``'bsn_round'``, or ``'bsn_bernoulli'``. If ``None``, no discretization."
    ),
    Kwarg(
        'encoder_weight_normalization',
        False,
        bool,
        "Apply weight normalization to encoder. Ignored unless encoder is recurrent."
    ),
    Kwarg(
        'encoder_layer_normalization',
        False,
        bool,
        "Apply layer normalization to encoder. Ignored unless encoder is recurrent."
    ),
    Kwarg(
        'decoder_type',
        'rnn',
        str,
        "Decoder network to use. One of ``dense``, ``cnn``, or ``rnn``."
    ),
    Kwarg(
        'n_units_decoder',
        None,
        [int, str, None],
        "Number of units to use in non-final decoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_decoder** - 1 space-delimited integers, one for each layer in order from top to bottom, or ``None``, in which case the units will be equal to **k**."
    ),
    Kwarg(
        'n_layers_decoder',
        2,
        int,
        "Number of layers to use for decoder. Ignored if **decoder_type** is not ``dense``."
    ),
    Kwarg(
        'decoder_activation',
        None,
        [str, None],
        "Name of activation to use at the output of the decoder"
    ),
    Kwarg(
        'decoder_inner_activation',
        None,
        [str, None],
        "Name of activation to use for any internal layers of the decoder",
        aliases=['inner_activation']
    ),
    Kwarg(
        'decoder_recurrent_activation',
        'hard_sigmoid',
        [str, None],
        "Name of activation to use for recurrent activation in recurrent layers of the decoder. Ignored if decoder is not recurrent.",
        aliases=['recurrent_activation']
    ),
    Kwarg(
        'conv_kernel_size',
        3,
        int,
        "Size of kernel to use in convolutional layers. Ignored if no residual layers in the model."
    ),
    Kwarg(
        'encoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal encoder layers as residual layers with **resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),
    Kwarg(
        'decoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal decode layers as residual layers with **resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),
    Kwarg(
        'batch_normalize_encodings',
        False,
        bool,
        "Batch normalize latent segment encodings."
    ),
    Kwarg(
        'encoder_batch_normalization_decay',
        0.9,
        [float, None],
        "Decay rate to use for batch normalization in internal encoder layers. If ``None``, no batch normalization.",
        aliases=['batch_normalization_decay']
    ),
    Kwarg(
        'decoder_batch_normalization_decay',
        0.9,
        [float, None],
        "Decay rate to use for batch normalization in internal decoder layers. If ``None``, no batch normalization.",
        aliases=['batch_normalization_decay']
    ),
    Kwarg(
        'pad_seqs',
        True,
        bool,
        "Whether to pad inputs and targets out to a fixed temporal dimensionality. Necessary in order to use minibatches larger than 1. If ``True``, sequences are padded and submitted to the network as batch arrays. If ``False``, sequences are not padded and are submitted to the network as minibatches of 1."
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
        'max_len',
        None,
        [int, None],
        "Maximum sequence length. If ``None``, no maximum length imposed."
    ),
    Kwarg(
        'resample_inputs',
        None,
        [int, None],
        "Resample inputs to fixed length **resample** timesteps. If ``None``, no input resampling."
    ),
    Kwarg(
        'resample_outputs',
        None,
        [int, None],
        "Resample targets to fixed length **resample** timesteps. If ``None``, no output resampling."
    ),
    Kwarg(
        'reverse_targets',
        True,
        bool,
        "Reverse the temporal dimension of the reconstruction targets."
    ),
    Kwarg(
        'reconstruct_deltas',
        False,
        bool,
        "Include derivatives in the reconstruction targets."
    ),
    Kwarg(
        'normalize_data',
        False,
        bool,
        "Normalize utterances to the range :math:`[0, 1]`. Mutually exclusive with **center_data**."
    ),
    Kwarg(
        'center_data',
        False,
        bool,
        "Center data about its mean. Mutually exclusive with **normalize_data**."
    ),
    Kwarg(
        'constrain_output',
        True,
        bool,
        "Use an output model constrained to :math:`[0, 1]` (sigmoid if MLE and LogitNormal if Bayesian). Otherwise, use linear/normal output. Requires **normalize_data* to be ``True``."
    ),
    Kwarg(
        'dtw_gamma',
        None,
        [float, None],
        "Smoothing parameter to use for soft-DTW objective. If ``Nonw``, do not use soft-DTW."
    ),
    Kwarg(
        'n_timesteps_input',
        None,
        [int, None],
        "Number of timesteps present in the input data. If ``None``, inferred from data, possibly with different values between batches."
    ),
    Kwarg(
        'n_timesteps_output',
        None,
        [int, None],
        "Number of timesteps present in the target data. If ``None``, inferred from data, possibly with different values between batches."
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
        1e-3,
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
        0.,
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
        1.,
        float,
        "coefficient by which to decay the learning rate every ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_steps',
        1,
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
        'entropy_regularizer_scale',
        None,
        [float, None],
        "Scale of regularizer on binary entropy. If ``None``, no entropy regularization."
    ),
    Kwarg(
        'boundary_regularizer_scale',
        None,
        [float, None],
        "Scale of regularizer on boundary activations. If ``None``, no boundary regularization."
    ),
    Kwarg(
        'lm_scale',
        None,
        [float, None],
        "Scale of encoder language modeling objective in the loss function. If ``None`` or 0, no language modeling objective is used."
    ),
    Kwarg(
        'segment_encoding_correspondence_regularizer_scale',
        None,
        [float, None],
        "Scale of regularizer encouraging correspondence between segment encodings in the encoder and segment decodings in the decoder. Only used if the encoder and decoder have identical numbers of layers with identical numbers of units in each layer. If ``None``, no regularization for segment encoding correspondence."
    ),
    Kwarg(
        'input_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop input data to the encoder. If ``None``, no input dropout."
    ),
    Kwarg(
        'ema_decay',
        None,
        [float, None],
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
        'segment_eval_freq',
        1,
        int,
        "Frequency (in iterations) with which to evaluate segmentations."
    ),
    Kwarg(
        'log_graph',
        False,
        bool,
        "Log the network graph to Tensorboard"
    ),
    Kwarg(
        'label_map_file',
        None,
        [str, None],
        "Path to CSV file mapping segment labels to strings to use in plots. Must contain text columns named 'source' and 'target'. If ``None``, source labels will be used."
    ),
    Kwarg(
        'keep_plot_history',
        False,
        bool,
        "Keep plots from each checkpoint of a run, which can help visualize learning trajectories but can also consume a lot of disk space. If ``False``, only the most recent plot of each type is kept."
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
