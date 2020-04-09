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
                        except ValueError:
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

    # Global
    Kwarg(
        'outdir',
        './dnnseg_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
    ),
    Kwarg(
        'task',
        'segmenter',
        str,
        "Task to perform. One of ``['classifier', 'segmenter']``."
    ),
    Kwarg(
        'streaming',
        False,
        bool,
        "Whether to train in streaming mode. If ``True``, past and/or future acoustic features are reconstructed from each frame in the input. If ``False``, VAD regions are pre-segmented and reconstructed from their final state. Ignored unless **task** is ``'segmenter'``."
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

    # Data
    Kwarg(
        'data_type',
        'acoustic',
        str,
        "Type of data to process (one of [``'acoustic'``, ``'text'``]."
    ),
    Kwarg(
        'filter_type',
        'mfcc',
        str,
        "Spectral filter to use (one of [``'mfcc'``, ``'cochleagram'``]. Ignored unless **data_type** is ``acoustic``."
    ),
    Kwarg(
        'segtype',
        'vad',
        str,
        'Utterance-level segmentation type used to chunk audio input (one of ["vad", "wrd", "phn"]).'
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
        'data_normalization',
        None,
        [str, None],
        "Normalization to apply to data as a preprocess. One of ``center`` (subtract the mean), ``standardize`` (Z-transform), ``range`` (divide by the range so values are in [0,1]), ``sum`` (divide by the sum so values sum to 1), ``scale##`` (where ## is a float by which to scale the response), and any norm type supported by ``np.linalg.norm`` (for example, using ``2`` yields a 2-norm, under which input vectors differ only by their angle). If ``None``, no data normalization."
    ),
    Kwarg(
        'reduction_axis',
        'time',
        str,
        "Reduction axis to use for data normalization. One of ``['time', 'freq', 'both']``."
    ),
    Kwarg(
        'use_normalization_mask',
        False,
        bool,
        "Whether to normalize over entire file or just VAD regions."
    ),
    Kwarg(
        'constrain_output',
        False,
        bool,
        "Use an output model constrained to :math:`[0, 1]` (sigmoid with cross-entropy loss if MLE and LogitNormal if Bayesian). Otherwise, use linear/normal output. Ignored unless **normalize_data* is ``True``."
    ),
    Kwarg(
        'pad_seqs',
        True,
        bool,
        "Whether to pad inputs and targets out to a fixed temporal dimensionality. Necessary in order to use minibatches larger than 1. If ``True``, sequences are padded and submitted to the network as batch arrays. If ``False``, sequences are not padded and are submitted to the network as minibatches of 1."
    ),
    Kwarg(
        'mask_padding',
        True,
        bool,
        "Mask padding frames in reconstruction targets so that they are ignored in gradient updates."
    ),
    Kwarg(
        'min_len',
        None,
        [int, None],
        "Minimum sequence length. If ``None``, defaults to **max_len**."
    ),
    Kwarg(
        'max_len',
        None,
        [int, None],
        "Maximum sequence length. If ``None``, no maximum length imposed."
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
        'resample_inputs',
        None,
        [int, None],
        "Resample inputs to fixed length **resample_inputs** timesteps. If ``None``, no input resampling."
    ),
    Kwarg(
        'resample_targets_bwd',
        None,
        [int, None],
        "Resample backward targets to fixed length **resample_targets_bwd** timesteps. If ``None``, no backward target resampling.",
        aliases=['resample_targets']
    ),
    Kwarg(
        'resample_targets_fwd',
        None,
        [int, None],
        "Resample forward targets to fixed length **resample_targets_fwd** timesteps. If ``None``, no forward target resampling.",
        aliases=['resample_targets']
    ),

    # Input model
    Kwarg(
        'input_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization of inputs. If ``None``, no batch normalization."
    ),

    # Speaker model
    Kwarg(
        'speaker_emb_dim',
        None,
        [int, None],
        "Train a **speaker_emb_dim** dimensional embedding of the speaker ID to each acoustic frame and to the utterance embedding. If ``None`` or ``0``, no speaker embedding used."
    ),
    Kwarg(
        'append_speaker_emb_to_inputs',
        False,
        bool,
        "Concatenate speaker embedding to inputs to encoder. Ignored if **speaker_emb_dim** is ``None`` or ``0``."
    ),
    Kwarg(
        'append_speaker_emb_to_decoder_inputs',
        False,
        bool,
        "Concatenate speaker embedding to inputs to decoder. Ignored if **speaker_emb_dim** is ``None`` or ``0``."
    ),
    Kwarg(
        'speaker_adversarial_gradient_scale',
        None,
        [float, None],
        "Scale of adversarial loss for residualizing speaker information out of the encoder. Ignored unless **speaker_emb_dim** is ``True``. If ``None``, no speaker adversarial training.",
        aliases=['adversarial_gradient_scale']
    ),
    Kwarg(
        'n_layers_speaker_decoder',
        2,
        [int, None],
        "Number of layers to use for speaker decoder. If ``None``, inferred from length of **n_units_speaker_decoder**."
    ),
    Kwarg(
        'n_units_speaker_decoder',
        256,
        [int, str, None],
        "Number of units to use in speaker decoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_speaker_decoder** - 1 space-delimited integers, one for each layer in order from top to bottom. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value."
    ),
    Kwarg(
        'speaker_decoder_activation_inner',
        'elu',
        [str, None],
        "Activation function to use for internal layers of speaker decoder."
    ),
    Kwarg(
        'speaker_revnet_n_layers',
        None,
        [int, None],
        "Number of layers in RevNet projection of spectral features. If ``None``, no RevNet projection.",
        aliases=['revnet_n_layers']
    ),
    Kwarg(
        'speaker_revnet_n_layers_inner',
        1,
        int,
        "Number of internal layers in each block of RevNet projection of spectral features. Ignored if **revnet_n_layers** is ``None``.",
        aliases=['revnet_n_layers_inner']
    ),
    Kwarg(
        'speaker_revnet_activation',
        'tanh',
        str,
        "Activation function to use in RevNet projection of spectral features. Ignored if **revnet_n_layers** is ``None``.",
        aliases=['revnet_activation']
    ),
    Kwarg(
        'speaker_revnet_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in RevNet projection of spectral features. If ``None``, no batch normalization. Ignored if **revnet_n_layers** is ``None``.",
        aliases=['batch_normalization_decay', 'revnet_batch_normalization_decay']
    ),

    # Passthru model
    Kwarg(
        'n_passthru_neurons',
        None,
        [int, None],
        "Number of passthru neurons to apply at the first layer of the encoder. Passthru neurons are dimensions of the underlying hidden state that get passed directly to the decoder without discretization or other constraints, and are adversarially regressed out of the rest of the hidden state. If ``None`` or ``0``, no passthru used."
    ),
    Kwarg(
        'passthru_adversarial_gradient_scale',
        None,
        [float, None],
        "Scale of adversarial loss for residualizing contents of passthru neurons out of the encoder. Ignored unless **speaker_emb_dim** is ``True``. If ``None``, no passthru adversarial training.",
        aliases=['adversarial_gradient_scale']
    ),
    Kwarg(
        'n_layers_passthru_decoder',
        2,
        [int, None],
        "Number of layers to use for passthru decoder. If ``None``, inferred from length of **n_units_passthru_decoder**."
    ),
    Kwarg(
        'n_units_passthru_decoder',
        256,
        [int, str, None],
        "Number of units to use in passthru decoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_passthru_decoder** - 1 space-delimited integers, one for each layer in order from top to bottom. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value."
    ),
    Kwarg(
        'passthru_decoder_activation_inner',
        'elu',
        [str, None],
        "Activation function to use for internal layers of passthru decoder."
    ),
    Kwarg(
        'emb_dim',
        None,
        [int, None],
        "Append **emb_dim** ELU-activated pass-through channels to the encoder output for capturing category-internal variation. If ``None`` or ``0``, no additional embedding dimensions."
    ),

    # Objective
    Kwarg(
        'predict_deltas',
        False,
        bool,
        "Include derivatives in the prediction targets."
    ),
    Kwarg(
        'dtw_gamma',
        None,
        [float, None],
        "Smoothing parameter to use for soft-DTW objective. If ``Nonw``, do not use soft-DTW."
    ),
    Kwarg(
        'l2_normalize_targets',
        False,
        bool,
        "Whether to L2 normalize decoder targets.",
    ),
    Kwarg(
        'residual_targets',
        False,
        bool,
        "Use the difference from one timestep to the next as the prediction target, rather than the raw data."
    ),
    Kwarg(
        'binary_classifier',
        True,
        bool,
        "Implement the classifier as a binary code in which categories can share bits. If ``False``, implements the classifier using independent categories."
    ),
    Kwarg(
        'predict_backward',
        True,
        bool,
        "Whether to predict backward (reconstruct previous inputs). Ignored unless **task** is ``'segmenter'`` and **streaming** is ``True``."
    ),
    Kwarg(
        'predict_forward',
        True,
        bool,
        "Whether to predict forward (predict future inputs). Ignored unless **task** is ``'segmenter'`` and **streaming** is ``True``."
    ),
    Kwarg(
        'window_len_bwd',
        50,
        int,
        "Length of backward-looking prediction targets (in frames). Ignored unless **task** is ``'segmenter'``, **streaming** is ``True``, and **predict_backward** is ``True``."
    ),
    Kwarg(
        'window_len_fwd',
        50,
        int,
        "Length of forward-looking prediction targets (in frames). Ignored unless **task** is ``'segmenter'``, **streaming** is ``True``, and **predict_forward** is ``True``."
    ),
    Kwarg(
        'reverse_targets',
        True,
        bool,
        "Reverse the temporal dimension of the reconstruction targets."
    ),
    Kwarg(
        'backprop_into_targets',
        False,
        bool,
        "Whether to backprop into prediction targets."
    ),
    Kwarg(
        'backprop_into_loss_weights',
        False,
        bool,
        "Whether to backprop into weight mask on losses (matters in ``masked_neighbor`` setting for LM loss)."
    ),
    Kwarg(
        'round_loss_weights',
        False,
        bool,
        "Whether to round (discretize) loss weights to {0,1}."
    ),
    Kwarg(
        'xent_state_predictions',
        False,
        bool,
        "Whether to convert encoder state values to probabilities and predict them using cross-entropy."
    ),

    # Correspondence AE
    Kwarg(
        'n_correspondence',
        None,
        [int, None],
        "Number of discovered segments to use to compute correspondence autoencoder auxiliary loss. If ``0`` or ``None``, do not use correpondence autoencoder."
    ),
    Kwarg(
        'sequential_cae',
        False,
        bool,
        'Whether to implement CAE loss using entire sequence. If ``False``, use the average over the sequence.'
    ),
    Kwarg(
        'n_layers_correspondence_decoder',
        2,
        [int, None],
        "Number of layers to use for correspondence decoder. If ``None``, inferred from length of **n_units_correspondence_decoder**."
    ),
    Kwarg(
        'n_units_correspondence_decoder',
        256,
        [int, str, None],
        "Number of units to use in correspondence decoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_correspondence_decoder** - 1 space-delimited integers, one for each layer in order from top to bottom. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value."
    ),
    Kwarg(
        'add_bottomup_correspondence_loss',
        False,
        bool,
        "Whether to additionally enforce a bottom-up (data to labels) correspondence loss."
    ),
    Kwarg(
        'correspondence_decoder_activation_inner',
        'elu',
        [str, None],
        "Activation function to use for internal layers of correspondence decoder."
    ),
    Kwarg(
        'resample_correspondence',
        25,
        int,
        "Number of timesteps to which correspondence autoencoder targets should be resampled. Ignored if **n_correspondence** is ``0`` or ``None``."
    ),
    Kwarg(
        'correspondence_start_step',
        1,
        int,
        "Step number (batch if **streaming** or iteration otherwise) at which to start minimizing correpondence autoencoder auxiliary loss. Ignored if **n_correspondence** is ``0`` or ``None``."
    ),
    Kwarg(
        'correspondence_loss_scale',
        None,
        [float, str, None],
        "Scale of layerwise encoder correspondence objective in the loss function. If a scalar is provided, it is applied uniformly to all layers. If ``None`` or 0, no correspondence objective is used."
    ),
    Kwarg(
        'correspondence_gradient_scale',
        None,
        [float, str, None],
        "Scale of gradients by layer to layerwise encoder correspondence objective. If a scalar is provided, it is applied uniformly to all layers. If ``None`` or 0, no correspondence objective is used."
    ),
    Kwarg(
        'correspondence_loss_implementation',
        3,
        int,
        "Implementation of correspondence AE loss. One of ``[1, 2, 3]``. Implementation 1: Take average of acoustics of saved segments, weighted by cosine similarity to current states. Implementation 2: Use acoustics of most similar saved segment to current state. Implementation 3: Take average of losses with respect to each saved segment, weighted by cosine similarity to the current state."
    ),
    Kwarg(
        'correspondence_live_targets',
        False,
        bool,
        "Whether to compute correspondence AE loss against targets sampled from segmentations generated for the current minibatch. If ``False``, correspondence targets are sampled from previous minibatch. When used, correspondence targets faithfully represent the current state of the network and losses backpropagate into the representations and boundaries of both segments in the pair, but computing losses is more computationally intensive because Fourier resampling of acoustic features must be performed inside the Tensorflow graph."
    ),
    Kwarg(
        'correspondence_n_timesteps',
        None,
        [int, None],
        "Number of timesteps per utterance to use for computing correspondence loss. If positive integer :math:`k`, use only the math:`k` most probable segmentation points per segmenter network. If ``0`` or ``None``, use all timesteps."
    ),
    Kwarg(
        'correspondence_alpha',
        1.,
        float,
        "Peakiness factor for correspondence AE loss. If ``1``, use cosine similarity weights directly. If ``< 1``, decrease peakiness of weights. If ``> 1``, increase peakiness of weights. Ignored unless **correspondence_implementation** is ``1``."
    ),

    # Language model
    Kwarg(
        'lm_loss_scale',
        None,
        [float, str, None],
        "Scale of layerwise encoder language modeling objective in the loss function. If a scalar is provided, it is applied uniformly to all layers. If ``None`` or 0, no language modeling objective is used."
    ),
    Kwarg(
        'lm_gradient_scale',
        1.,
        [float, str, None],
        "Scale of gradients by layer to layerwise encoder language modeling objective. If a scalar is provided, it is applied uniformly to all layers. If ``None`` or 0, no language modeling objective is used."
    ),
    Kwarg(
        'lm_target_gradient_scale',
        1.,
        [float, str, None],
        "Scale of gradients by layer to targets for layerwise encoder language modeling objective. If a scalar is provided, it is applied uniformly to all layers. If ``None`` or 0, no language modeling objective is used."
    ),
    Kwarg(
        'lm_loss_type',
        'masked_neighbors',
        str,
        "Type of LM loss. One of ``['neighbors', 'masked_neighbors', 'srn']``."
    ),
    Kwarg(
        'lm_masking_mode',
        'drop_masked',
        str,
        "Type of prediction to use in masked LM mode. One of ``['drop_masked', 'predict_at_boundaries', 'predict_everywhere']``. If ``drop_masked``, drop all non-boundary frames. If ``predict_at_masked``, predict non-boundary frames but only at boundaries. If ``predict_everywhere``, predict non-boundary frames from all timesteps. Ignored unless **lm_loss_type** is ``'masked_neighbors'``."
    ),
    Kwarg(
        'lm_order_fwd',
        1,
        int,
        "Order of forward language model (number of timesteps to predict)."
    ),
    Kwarg(
        'lm_order_bwd',
        0,
        int,
        "Order of backward language model (number of timesteps to predict)."
    ),
    Kwarg(
        'lm_use_upper',
        False,
        bool,
        "Whether to condition LM predictions on upper layers."
    ),
    Kwarg(
        'lm_decode_from_encoder_states',
        False,
        bool,
        "Whether to decode from encoder hidden states. If ``False``, decode from encoder features/embeddings."
    ),
    Kwarg(
        'lm_boundaries_as_attn',
        False,
        bool,
        "Whether to use boundaries as attention mask to admit upper layers in proportion to the probability that they are the lowest non-segmenting layer. Ignored unless **lm_use_upper** is ``True``."
    ),
    Kwarg(
        'scale_losses_by_boundaries',
        False,
        bool,
        "Whether to scale LM and CAE losses by the corresponding boundary decisions.",
    ),

    # Curriculum settings
    Kwarg(
        'curriculum_type',
        None,
        [str, None],
        "Type of curriculum learning decay to use. One of [``'hard'``, ``'exp'``, ``'sigmoid'``]. Type ``'hard'`` incrementally increases the number timesteps over which to compute losses. Type ``'exp'`` uses incrementally relaxed exponentially decaying weights on the losses by timestep. Type ``'sigmoid'`` uses the complement of a sigmoid function timesteps, with center shifting incrementally to the right. If ``None``, no curriculum learning."
    ),
    Kwarg(
        'curriculum_init',
        1.,
        float,
        "If **curriculum_type** is ``'hard'``, initial window length to use for curriculum learning. If **curriculum_type** is ``'exp'``, initial value for denominator of exponential decay rate. If **curriclum_type** is ``'sigmoid'``, slope of sigmoid function. Ignored if **curriculum_type** is ``None``."
    ),
    Kwarg(
        'curriculum_steps',
        1,
        int,
        "Number of steps (minibatches if **streaming** is ``True``, otherwise iterations) over which to execute a unit increase in window length (if **curriculum_type** is ``'hard'``), denominator of exponential decay rate (if **curriculum_type** is ``'exp'``), or shift of sigmoid center (if **curriculum_type** is ``'sigmoid'``). Ignored if **curriculum_type** is ``None``."
    ),

    # Regularization
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
        'segment_encoding_correspondence_regularizer_scale',
        None,
        [float, None],
        "Scale of regularizer encouraging correspondence between segment encodings in the encoder and segment decodings in the decoder. Only used if the encoder and decoder have identical numbers of layers with identical numbers of units in each layer. If ``None``, no regularization for segment encoding correspondence."
    ),

    # Encoder
    Kwarg(
        'n_layers_input_projection',
        None,
        [int, None],
        "Number of layers to use for input projection. If ``None``, inferred from length of **n_units_input_projection**."
    ),
    Kwarg(
        'n_units_input_projection',
        None,
        [int, str, None],
        "Number of units to use in input projection layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_input_projection** space-delimited integers, one for each layer in order from bottom to top. If ``None`` no input projection."
    ),
    Kwarg(
        'n_layers_pre_cnn',
        None,
        [int, None],
        "Number of layers to use for pre-encoder CNN. If ``None``, inferred from length of **n_layers_pre_cnn**."
    ),
    Kwarg(
        'n_units_pre_cnn',
        128,
        [int, str, None],
        "Number of units to use in pre-encoder CNN. Can be an ``int``, which will be used for all layers, a ``str`` with **n_units_pre_cnn** - 1 space-delimited integers, one for each layer in order from top to bottom. If ``None``, no pre-CNN."
    ),
    Kwarg(
        'n_layers_pre_rnn',
        None,
        [int, None],
        "Number of layers to use for pre-encoder RNN. If ``None``, inferred from length of **n_layers_pre_rnn**."
    ),
    Kwarg(
        'n_units_pre_rnn',
        128,
        [int, str, None],
        "Number of units to use in pre-encoder RNN. Can be an ``int``, which will be used for all layers, a ``str`` with **n_units_pre_rnn** - 1 space-delimited integers, one for each layer in order from top to bottom. If ``None``, no pre-RNN."
    ),
    Kwarg(
        'input_projection_activation_inner',
        'elu',
        [str, None],
        "Activation function to use for internal layers of input projection."
    ),
    Kwarg(
        'input_projection_activation',
        None,
        [str, None],
        "Activation function to use for input projection."
    ),
    Kwarg(
        'encoder_type',
        'rnn',
        str,
        "Encoder network to use. One of ``dense``, ``cnn``, or ``rnn``."
    ),
    Kwarg(
        'encoder_use_bias',
        True,
        bool,
        "Whether to include bias units in encoder layers."
    ),
    Kwarg(
        'encoder_bptt',
        True,
        bool,
        "Backpropagate error through time in the HM-LSTM encoder.",
    ),
    Kwarg(
        'encoder_conv_kernel_size',
        3,
        int,
        "Size of kernel to use in convolutional encoder layers. Ignored if no convolutional encoder layers in the model.",
        aliases=['conv_kernel_size']
    ),
    Kwarg(
        'n_layers_encoder',
        None,
        [int, None],
        "Number of layers to use for encoder. If ``None``, inferred from length of **n_units_encoder**."
    ),
    Kwarg(
        'n_units_encoder',
        None,
        [int, str, None],
        "Number of units to use in layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_encoder** space-delimited integers, one for each layer in order from bottom to top. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value."
    ),
    Kwarg(
        'n_features_encoder',
        None,
        [str, None],
        "Number of features to use in encoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_encoder** space-delimited integers, one for each layer in order from bottom to top. If ``None``, encoder units will be used directly as features without transformation. Ignored unless the encoder is an HM-LSTM."
    ),
    Kwarg(
        'n_embdims_encoder',
        None,
        [str, None],
        "Number of embedding dimensions to use in encoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_encoder** space-delimited integers, one for each layer in order from bottom to top. If ``None``, encoder bits will be used directly as features without embedding. Ignored unless the encoder is an HM-LSTM."
    ),
    Kwarg(
        'hmlstm_kernel_depth',
        1,
        int,
        "Depth of deep kernel in HM-LSTM transition model."
    ),
    Kwarg(
        'hmlstm_featurizer_kernel_depth',
        1,
        int,
        "Depth of deep kernel in HM-LSTM featurizer."
    ),
    Kwarg(
        'hmlstm_prefinal_mode',
        'max',
        str,
        "Mode for choosing the number of hidden units in pre-final layers of HM-LSTM's with deep transitions. One of ``['in', 'out', 'max']``, for number input dimensions, number of output dimensions, and max of input and output dimensions, respectively. Ignored unless **hmlstm_kernel_depth** > 1."
    ),
    Kwarg(
        'encoder_activation',
        None,
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
        'sigmoid',
        [str, None],
        "Name of activation to use for recurrent activation in recurrent layers of the encoder. Ignored if encoder is not recurrent.",
        aliases=['recurrent_activation']
    ),
    Kwarg(
        'encoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal encoder layers as residual layers with **resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),
    Kwarg(
        'encoder_renormalize_preactivations',
        False,
        bool,
        "Whether to compute HMLSTM preactivations as an average weighted by segmentation decisions."
    ),
    Kwarg(
        'encoder_append_previous_features',
        False,
        bool,
        "Whether to append features from the preceding segment to inputs to the HMLSTM recurrence. Ignored unless the decoder is an HMLSTM and encoder_num_features is defined for the current layer."
    ),
    Kwarg(
        'encoder_append_seg_len',
        False,
        bool,
        "Whether to append incoming segment lengths to non-initial HMLSTM layers. Ignored unless the decoder is an HMLSTM."
    ),

    # Encoder boundaries
    Kwarg(
        'encoder_boundary_implementation',
        2,
        int,
        "Implementation to use for HM-LSTM encoder boundary neuron. If ``1``, use a dedicated cell of the hidden state. If ``2``, use a dense kernel over the hidden state.",
    ),
    Kwarg(
        'nested_boundaries',
        False,
        bool,
        "Whether to mask boundaries using the boundaries from the layer below."
    ),
    Kwarg(
        'encoder_boundary_activation',
        'sigmoid',
        [str, None],
        "Name of activation to use for boundary activation in the HM-LSTM encoder. Ignored if encoder is not an HM-LSTM.",
        aliases=['boundary_activation']
    ),
    Kwarg(
        'encoder_prefinal_activation',
        'elu',
        [str, None],
        "Name of activation to use for prefinal layers in an HM-LSTM encoder with deep transitions. Ignored if encoder is not an HM-LSTM or if **hmlstm_kernel_depth** < 2.",
        aliases=['boundary_activation']
    ),
    Kwarg(
        'oracle_boundaries',
        None,
        [str, None],
        "Type of boundary to use for oracle evaluation (one of ``['vad', 'phn', 'wrd', 'rnd', None]``). If ``None``, do not use oracle boundaries. Ignored unless **task** is ``'segmenter'``."
    ),
    Kwarg(
        'random_oracle_segmentation_rate',
        None,
        [float, None],
        "Rate of segmentation to use if **oracle_boundaries** is ``rnd``."
    ),
    Kwarg(
        'encoder_force_vad_boundaries',
        True,
        bool,
        "Whether to force segmentation probabilities to 1 at the ends of VAD regions."
    ),
    Kwarg(
        'neurons_per_boundary',
        1,
        int,
        "Number of boundary neurons to use in segmenter."
    ),
    Kwarg(
        'feature_neuron_agg_fn',
        'logsumexp',
        str,
        "Name of aggregation function to use over feature neurons when **n_feature_neurons** > 1.",
        aliases=['neuron_agg_fn']
    ),
    Kwarg(
        'neurons_per_feature',
        1,
        int,
        "Number of per-feature neurons to use in encoder."
    ),
    Kwarg(
        'boundary_neuron_agg_fn',
        'logsumexp',
        str,
        "Name of aggregation function to use over boundary neurons when **n_boundary_neurons** > 1.",
        aliases=['neuron_agg_fn']
    ),
    Kwarg(
        'cumulative_boundary_prob',
        False,
        bool,
        "Whether to accumulate boundary probs over time (if not, full boundary prob is generated at each timestep)."
    ),
    Kwarg(
        'cumulative_feature_prob',
        False,
        bool,
        "Whether to accumulate feature probs over time (if not, full feature prob is generated at each timestep)."
    ),
    Kwarg(
        'forget_at_boundary',
        True,
        bool,
        "Whether to flush the cell state at boundaries, as done in Chung et al 17."
    ),
    Kwarg(
        'recurrent_at_forget',
        True,
        bool,
        "Whether to retain recurrent connection to previous hidden state over segment boundaries."
    ),

    # Encoder RevNet
    Kwarg(
        'encoder_revnet_n_layers',
        None,
        [int, None],
        "Number of layers in RevNet projection of layerwise encoder inputs. If ``None``, no RevNet projection.",
        aliases=['revnet_n_layers']
    ),
    Kwarg(
        'encoder_revnet_n_layers_inner',
        1,
        int,
        "Number of internal layers in each block of RevNet projection of layerwise encoder inputs. Ignored if **revnet_n_layers** is ``None``.",
        aliases=['revnet_n_layers_inner']
    ),
    Kwarg(
        'encoder_revnet_activation',
        'elu',
        str,
        "Activation function to use in RevNet projection of layerwise encoder inputs. Ignored if **revnet_n_layers** is ``None``.",
        aliases=['revnet_activation']
    ),
    Kwarg(
        'encoder_revnet_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in RevNet projection of layerwise encoder inputs. If ``None``, no batch normalization. Ignored if **revnet_n_layers** is ``None``.",
        aliases=['batch_normalization_decay', 'revnet_batch_normalization_decay']
    ),

    # Encoder discretization
    Kwarg(
        'sample_at_train',
        True,
        bool,
        "Use Bernoulli sampling (rather than rounding) for BSNs during training phase. Ignored unless model contains BSNs."
    ),
    Kwarg(
        'sample_at_eval',
        False,
        bool,
        "Use Bernoulli sampling (rather than rounding) for BSNs during evaluation phase. Ignored unless model contains BSNs."
    ),
    Kwarg(
        'min_discretization_prob',
        None,
        [float, None],
        "Minimum probability of discretizing. If ``None``, always discretize discretized variables."
    ),
    Kwarg(
        'trainable_self_discretization',
        True,
        bool,
        "Whether to allow gradients into the decision to discretize. Ignored if **min_discretization_prob** is ``None``."
    ),
    Kwarg(
        'slope_annealing_max',
        None,
        [float, None],
        "Maximum allowed value of the slope annealing coefficient. If ``None``, no maximum will be enforced."
    ),

    # Encoder boundary discretization
    Kwarg(
        'encoder_boundary_discretizer',
        None,
        [str, None],
        "Discretization function to apply to encoder boundary activations, currently only ``None`` and ``bsn`` supported. If ``None``, no discretization."
    ),
    Kwarg(
        'boundary_slope_annealing_rate',
        None,
        [float, None],
        "Whether to anneal the slopes of the boundary activations.",
        aliases=['slope_annealing_rate']
    ),

    # Encoder state discretization
    Kwarg(
        'encoder_state_discretizer',
        None,
        [str, None],
        "Discretization function to apply to encoder hidden states, currently only ``None`` and ``bsn`` supported. If ``None``, no discretization."
    ),
    Kwarg(
        'encoder_discretize_state_at_boundary',
        False,
        bool,
        "Discretize state at boundary only. Otherwise, encoder state is fully discretized. Ignored if **encoder_state_discretizer** is ``None``."
    ),
    Kwarg(
        'encoder_discretize_final',
        False,
        bool,
        "Whether to discretize the final layer. Ignored if **encoder_state_discretizer** is ``None``."
    ),
    Kwarg(
        'state_slope_annealing_rate',
        None,
        [float, None],
        "Whether to anneal the slopes of the hidden state activations.",
        aliases=['slope_annealing_rate']
    ),

    # Encoder normalization
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
        'encoder_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in internal encoder layers. If ``None``, no batch normalization.",
        aliases=['batch_normalization_decay']
    ),
    Kwarg(
        'batch_normalize_encodings',
        False,
        bool,
        "Batch normalize latent segment encodings."
    ),
    Kwarg(
        'encoder_l2_normalize_states',
        False,
        bool,
        "Whether to L2 normalize encoder states.",
    ),

    # Encoder regularization
    Kwarg(
        'encoder_weight_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of encoder weight regularization. If ``float``, scale of encoder L2 weight regularization. If ``None``, no encoder weight regularization."
    ),
    Kwarg(
        'encoder_state_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of encoder state regularization. If ``float``, scale of encoder L2 state regularization. If ``None``, no encoder state regularization."
    ),
    Kwarg(
        'encoder_feature_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of encoder feature regularization. If ``float``, scale of encoder L2 feature regularization. If ``None``, no encoder feature regularization."
    ),
    Kwarg(
        'encoder_bitwise_feature_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of encoder bitwise feature regularization. If ``float``, scale of encoder L2 bitwise feature regularization. If ``None``, no encoder bitwise feature regularization."
    ),
    Kwarg(
        'encoder_feature_similarity_regularizer_scale',
        None,
        [float, None],
        "Scale of penalty on similarity of consecutive encodings. If ``None``, no feature similarity penalty."
    ),
    Kwarg(
        'encoder_feature_similarity_regularizer_shape',
        1,
        float,
        "Shape parameter for exponential distribution used to compute the feature similarity regularizer."
    ),
    Kwarg(
        'encoder_seglen_regularizer_scale',
        None,
        [float, str, None],
        "Scale of segment length regularizer. If a scalar is provided, it is applied uniformly to all layers. If ``None``, no segment length regularization."
    ),
    Kwarg(
        'encoder_seglen_regularizer_shape',
        1.,
        float,
        "Shape parameter for exponential distribution used to compute the segment length regularizer."
    ),
    Kwarg(
        'encoder_cell_proposal_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of encoder cell proposal regularization. If ``float``, scale of encoder L2 cell proposal regularization. If ``None``, no encoder cell proposal regularization."
    ),
    Kwarg(
        'encoder_dropout',
        None,
        [float, None],
        "Dropout rate to use in the encoder",
    ),
    Kwarg(
        'temporal_dropout_rate',
        None,
        [str, float, None],
        "Rate at which to drop timesteps to the encoder during training. If ``None``, no temporal dropout."
    ),
    Kwarg(
        'temporal_dropout_plug_lm',
        False,
        bool,
        "Whether to plug language model predictions into dropped timesteps when temporal dropout is on."
    ),
    Kwarg(
        'encoder_bottomup_noise_level',
        None,
        [float, None],
        "Standard deviation of Gaussian 'whiteout' noise to inject into the encoder bottom-up inputs at each layer.",
        aliases=['encoder_input_noise_level', 'encoder_bottomup_noise_sd', 'encoder_input_noise_sd']
    ),
    Kwarg(
        'encoder_recurrent_noise_level',
        None,
        [float, None],
        "Standard deviation of Gaussian 'whiteout' noise to inject into the encoder recurrent inputs at each layer.",
        aliases=['encoder_recurrent_noise_sd']
    ),
    Kwarg(
        'encoder_topdown_noise_level',
        None,
        [float, None],
        "Standard deviation of Gaussian 'whiteout' noise to inject into the encoder top-down inputs at each layer.",
        aliases=['encoder_input_noise_level', 'encoder_topdown_noise_sd', 'encoder_input_noise_sd']
    ),
    Kwarg(
        'encoder_boundary_noise_level',
        None,
        [float, None],
        "Standard deviation of Gaussian 'whiteout' noise to inject into logits of boundary probabilities during training.",
        aliases=['encoder_boundary_noise_sd']
    ),
    Kwarg(
        'encoder_state_noise_level',
        None,
        [float, None],
        "Standard deviation of Gaussian 'whiteout' noise to inject into the pre-gated encoder outputs.",
        aliases=['encoder_state_noise_sd']
    ),
    Kwarg(
        'encoder_feature_noise_level',
        None,
        [float, None],
        "Standard deviation of Gaussian 'whiteout' noise to inject into the encoder feature pre-activations.",
        aliases=['encoder_feature_noise_sd']
    ),
    Kwarg(
        'boundary_rate_extremeness_regularizer_scale',
        None,
        [float, None],
        "Scale of penalty on extreme boundary rates (very low or very high). If ``None``, no boundary rate extremeness penalty."
    ),
    Kwarg(
        'boundary_rate_extremeness_regularizer_shape',
        0.1,
        float,
        "Shape parameter ``a in [0,1]`` on boundary rate ``p``, such that the boundary rate extremeness penalty is proportional to ``Beta(a, a).pdf(p)``. The lower the shape value, the more the penalty is pushed to the edges (high or low values of p)."
    ),
    Kwarg(
        'boundary_prob_extremeness_regularizer_scale',
        None,
        [float, None],
        "Scale of penalty on extreme boundary probs (very low or very high). If ``None``, no boundary prob extremeness penalty."
    ),
    Kwarg(
        'boundary_prob_extremeness_regularizer_shape',
        0.1,
        float,
        "Shape parameter ``a in [0,1]`` on mean boundary prob ``p``, such that the boundary prob extremeness penalty is proportional to ``Beta(a, a).pdf(p)``. The lower the shape value, the more the penalty is pushed to the edges (high or low values of p)."
    ),
    Kwarg(
        'feature_rate_extremeness_regularizer_scale',
        None,
        [float, None],
        "Scale of penalty on extreme feature rates (very low or very high). If ``None``, no feature rate extremeness penalty."
    ),
    Kwarg(
        'feature_rate_extremeness_regularizer_shape',
        0.1,
        float,
        "Shape parameter ``a in [0,1]`` on feature rate ``p``, such that the feature rate extremeness penalty is proportional to ``Beta(a, a).pdf(p)``. The lower the shape value, the more the penalty is pushed to the edges (high or low values of p)."
    ),
    Kwarg(
        'entropy_regularizer_scale',
        None,
        [float, None],
        "Scale of regularizer on binary entropy. If ``None``, no entropy regularization."
    ),
    Kwarg(
        'boundary_prob_regularizer_scale',
        None,
        [float, None],
        "Scale of regularizer on boundary activation probabilities. If ``None``, no boundary probability regularization."
    ),
    Kwarg(
        'boundary_regularizer_scale',
        None,
        [float, None],
        "Scale of regularizer on boundary activations. If ``None``, no boundary regularization."
    ),
    Kwarg(
        'segment_at_peaks',
        False,
        bool,
        'Re-extracting segment boundaries at peak values of the segmentation probability vector. Does not affect internal segmentation behavior of the HMLSTM cell but changes selection of segments for the correspondence AE.'
    ),
    Kwarg(
        'boundary_prob_discretization_threshold',
        None,
        [float, None],
        'Minimum value that boundary probabilities must exceed in order to be eligible candidates for a discrete segmentation boundary. Has no effect unless **segment_at_peaks** is ``True``.'
    ),
    Kwarg(
        'boundary_prob_smoothing',
        None,
        [str, None],
        'Post-process segmentations by smoothing them using function defined by underscore_delimited string, where the first element is the type of smooth and all subsequent elements are positional arguments. One of ["rbf_<order>_<penalty>", "ema_<decay>", "dema_<decay>", "wma_<width>"], where "rbf", "ema", "dema", and "wma" are (respectively) radial basis function, exponential moving average, double exponential moving average, and weighted moving average.  If ``None``, no smoothing. Has no effect unless **segment_at_peaks** is ``True``.'
    ),

    # Decoder
    Kwarg(
        'decoder_type',
        'rnn',
        str,
        "Decoder network to use. One of ``dense``, ``cnn``, or ``rnn``."
    ),
    Kwarg(
        'decoder_concatenate_hidden_states',
        False,
        bool,
        "Whether to concatenate the hidden states from all encoder layers as input to the decoder. If ``False``, only the hidden state from the final layer will be used."
    ),
    Kwarg(
        'decoder_conv_kernel_size',
        3,
        int,
        "Size of kernel to use in convolutional decoder layers. Ignored if no convolutional decoder layers in the model."
    ),
    Kwarg(
        'n_layers_decoder',
        2,
        [int, None],
        "Number of layers to use for decoder. If ``None``, inferred from length of **n_units_decoder**."
    ),
    Kwarg(
        'n_units_decoder',
        None,
        [int, str, None],
        "Number of units to use in decoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_decoder** - 1 space-delimited integers, one for each layer in order from top to bottom. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value."
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
        'sigmoid',
        [str, None],
        "Name of activation to use for recurrent activation in recurrent layers of the decoder. Ignored if decoder is not recurrent.",
        aliases=['recurrent_activation']
    ),
    Kwarg(
        'decoder_n_query_units',
        None,
        [int, None],
        "Number of dimensions in key/query component of attentional decoder. If ``None``, defaults to the dimensionality of the input key matrix."
    ),
    Kwarg(
        'decoder_project_attn_keys',
        True,
        bool,
        "Whether to project keys for computing attention weights."
    ),
    Kwarg(
        'decoder_use_gold_attn_keys_at_train',
        False,
        bool,
        "Whether to use gold or reconstructed segment IDs as keys for computing attention weights during training."
    ),
    Kwarg(
        'decoder_use_gold_attn_keys_at_eval',
        False,
        bool,
        "Whether to use gold or reconstructed segment IDs as keys for computing attention weights during evaluation."
    ),
    Kwarg(
        'decoder_positional_attn_keys',
        False,
        bool,
        "Whether to use positional encodings as attention keys. If ``False``, segment encodings are used."
    ),
    Kwarg(
        'decoder_gaussian_attn',
        False,
        bool,
        "Whether to a trainable Gaussian filter as an attention distribution. Otherwise, use scaled dot attn over segmental or positional keys."
    ),
    Kwarg(
        'backprop_into_attn_keys',
        True,
        bool,
        "Whether to backprop into seq2seq decoder keys."
    ),
    Kwarg(
        'decoder_encode_keys',
        False,
        bool,
        "Whether to run an RNN encoder over the attention keys in order to compute the initial state for the seq2seq decoder. If ``False``, the encoder hidden state will be used for state initialization."
    ),
    Kwarg(
        'decoder_discretize_attn_keys',
        False,
        bool,
        "Whether to discretize decoder attention keys in Seq2Seq decoder."
    ),
    Kwarg(
        'decoder_discretize_refeed',
        False,
        bool,
        "Whether to discretize previous decoder outputs as inputs to Seq2Seq decoder. At lowest layer, no discretization if acoustic mode, non-differentiable argmax discretization if text mode. For higher layers, discretize using differentiable binary stochastic neurons."
    ),
    Kwarg(
        'decoder_initialize_state_from_above',
        True,
        bool,
        "Whether to initialize the seq2seq decoder state using the layer above (which has seen the current layer's label sequence) or the current layer (which has seen the targets)."
    ),
    Kwarg(
        'backprop_into_refeed',
        True,
        bool,
        "Whether to backpropagate into decoder outputs as inputs to Seq2Seq decoder."
    ),
    Kwarg(
        'decoder_add_positional_encoding_to_top',
        False,
        bool,
        "Whether to add positional encoding to top layer decoder in the seq2seq setting, since this layer has no attention keys."
    ),
    Kwarg(
        'decoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal decode layers as residual layers with **resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
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

    # Decoder normalization
    Kwarg(
        'decoder_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in internal decoder layers. If ``None``, no batch normalization.",
        aliases=['batch_normalization_decay']
    ),

    # Decoder regularization
    Kwarg(
        'decoder_dropout',
        None,
        [float, None],
        "Dropout rate to use in the decoder",
    ),
    
    # Decoder projection
    Kwarg(
        'n_layers_decoder_input_projection',
        None,
        [int, None],
        "Number of hidden layers to use for projection of decoder inputs. If ``None``, inferred from ``n_units_decoder_input_projection``."
    ),
    Kwarg(
        'n_units_decoder_input_projection',
        None,
        [int, str, None],
        "Number of units in hidden layers to use for projection of decoder inputs. If ``None``, no projection."
    ),
    Kwarg(
        'decoder_input_projection_activation_inner',
        'elu',
        [str, None],
        "Name of activation to use for prefinal layers in projection function of decoder inputs.",
    ),
    Kwarg(
        'decoder_input_projection_activation',
        'elu',
        [str, None],
        "Name of activation to use for final layer in projection function of decoder inputs.",
    ),

    # Decoder positional encoding
    Kwarg(
        'decoder_hidden_state_expansion_type',
        'tile',
        str,
        "Technique for expanding the decoder inputs over time. One of [``tile``, ``dense``], where ``tile`` tiles the state over time while ``dense`` applies and reshapes a dense layer with ``n_feats * n_timesteps`` outputs."
    ),
    Kwarg(
        'decoder_positional_encoding_type',
        None,
        [None, str],
        "Technique for representing time to the decoder. One of [``transformer_pe``, ``periodic``, ``weights``], where ``transformer_pe`` uses the Transformer positional encoding, ``periodic`` uses **decoder_positional_encoding_units** sin and cosine waves (with more high-frequency components than Transformer), and ``weights`` uses trainable vectors of **decoder_positional_encoding_units**, one for each timestep. If ``None``, no temporal encoding."
    ),
    Kwarg(
        'decoder_positional_encoding_units',
        32,
        [int, None],
        "Number of dimensions per timestep to use for the temporal input to the decoder."
    ),
    Kwarg(
        'decoder_positional_encoding_transform',
        None,
        [str, None],
        "Technique for transforming the base temporal input to the decoder. One of [``None``, ``dense``, ``rnn``, ``cnn``]. If ``None``, no transform."
    ),
    Kwarg(
        'decoder_positional_encoding_activation',
        None,
        [str, None],
        "Activation function for temporal encoding, or linear activation if ``None``."
    ),
    Kwarg(
        'decoder_positional_encoding_as_mask',
        False,
        bool,
        "Whether to use the temporal encoding as a mask. If ``True``, temporal encoding will be conformable to input encoder state and sigmoid activated, serving as soft attention over the expanded hidden state. If ``False``, the temporal encoding is dimensionality **decoder_positional_encoding_units** and is concatenated to the expanded hidden state features."
    ),
    Kwarg(
        'decoder_positional_encoding_lock_to_data',
        True,
        bool,
        "Whether to lock the positional encoding to the frame rate of the data. If ``True``, positional encodings will reflect the distance from decoder start in number of input frames. Otherwise, positional embeddings will reflect the distance from decoder start in number of segments. Affects higher layer language modeling losses."
    ),

    # Optimization
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
        'use_jtps',
        False,
        bool,
        "Whether to modify the base optimizer using JTPS. If ``False``, runs a baseline model. If ``True``, runs a test model."
    ),
    Kwarg(
        'max_global_gradient_norm',
        None,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no gradient clipping.'
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
        0.001,
        [float, str],
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
        'lr_decay_iteration_power',
        1,
        float,
        "Power to which the iteration number ``t`` should be raised when computing the learning rate decay."
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
        'n_samples',
        1,
        int,
        "Number of samples to take per training point. Do not set to > 1 unless model contains stochastic decisions (e.g. Bernoulli-distributed binary stochastic neurons)."
    ),
    Kwarg(
        'weight_update_mode',
        'all',
        [str, float],
        "Weight update mode, one of ``'all'``, ``'boundary'``, ``'state'``, or ``float``."
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
        'n_pretrain_steps',
        0,
        int,
        "Number of steps (minibatches if **streaming** is ``True``, otherwise iterations) during which to pre-train the decoder without backpropagating into the encoder."
    ),
    Kwarg(
        'encoder_decoder_step_ratio',
        None,
        [float, None],
        "Ratio of encoder training steps to decoder training steps. If ``None``, or ``0``, jointly trained."
    ),
    Kwarg(
        'loss_normalization',
        None,
        [str, None],
        "Loss normalization method. If ``'layer'``, normalizes by layer. If ``'cell'``, renormalizes by cell. If ``None``, no renormalization."
    ),

    # Checkpoint
    Kwarg(
        'save_freq',
        1,
        int,
        "Frequency with which to save model checkpoints. If **streaming**, frequency is in minibatches and model is saved after each iteration; otherwise it's in iterations."
    ),
    Kwarg(
        'eval_freq',
        1,
        int,
        "Frequency with which to evaluate model. If **streaming**, frequency is in minibatches and model is evaluated after each iteration; otherwise it's in iterations."
    ),
    Kwarg(
        'log_freq',
        1,
        int,
        "Frequency with which to log summary data. If **streaming**, frequency is in minibatches and data is logged after each iteration; otherwise it's in iterations."
    ),
    Kwarg(
        'log_graph',
        False,
        bool,
        "Log the network graph to Tensorboard"
    ),

    # Visualization
    Kwarg(
        'label_map_file',
        None,
        [str, None],
        "Path to CSV file mapping segment labels to strings to use in plots. Must contain text columns named 'source' and 'target'. If ``None``, source labels will be used."
    ),
    Kwarg(
        'feature_map_file',
        None,
        [str, None],
        "Path to CSV file mapping segment labels to phonological distinctive features to use in plots. Must contain a text column named 'symbol', along with columns for any features of interest. If ``None``, no featural analysis will be performed."
    ),
    Kwarg(
        'plot_position_anchor',
        'input',
        str,
        "Type of temporal slice to use for plotting, one of ``['input', 'output']``. If ``'input'``, plot all predicted outputs from a single input timepoint. If ``'output'``, plot a single output position at all input timepoints."
    ),
    Kwarg(
        'plot_position_index',
        'mid',
        [int, str],
        "Position of index at which to plot. Can be a scalar or a space-delimited pair. If a scalar, the same index is used both forward and backward. If a pair, the first element is the backward index and the second element is the forward index. Also accepts the keyword ``'mid'``, which will use a point in the middle of the time series."
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


def dnnseg_kwarg_docstring():
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
