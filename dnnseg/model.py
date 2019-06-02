import sys
import os
import re
import math
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_mutual_info_score, fowlkes_mallows_score

from .backend import *
from .data import get_random_permutation, get_padded_lags, extract_segment_timestamps_batch, extract_states_at_timestamps_batch
from .kwargs import UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS
from .util import f_measure, pretty_print_seconds
from .plot import plot_acoustic_features, plot_label_histogram, plot_label_heatmap, plot_binary_unit_heatmap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

is_embedding_dimension = re.compile('d([0-9]+)')
regularizer = re.compile('([^_]+)(_([0-9]*\.?[0-9]*))?')


class AcousticEncoderDecoder(object):

    ############################################################
    # Initialization methods
    ############################################################

    _INITIALIZATION_KWARGS = UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for unsupervised word classifier. Bayesian and MLE implementations inherit from ``AcousticEncoderDecoder``.
        ``AcousticEncoderDecoder`` is not a complete implementation and cannot be instantiated.

    """
    _doc_args = """
        :param k: ``int``; dimensionality of classifier.
        :param train_data: ``AcousticDataset`` object; training data.
    \n"""
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (
                                 x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value)
                             for
                             x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    def __new__(cls, *args, **kwargs):
        if cls is AcousticEncoderDecoder:
            raise TypeError("UnsupervisedWordClassifier is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, k, train_data, **kwargs):

        self.k = k
        for kwarg in AcousticEncoderDecoder._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))
        if self.speaker_emb_dim:
            self.speaker_list = train_data.segments().speaker.unique()
        else:
            self.speaker_list = []

        self.plot_ix = None

        self._initialize_session()

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)
        self.UINT_TF = getattr(np, 'u' + self.int_type)
        self.UINT_NP = getattr(tf, 'u' + self.int_type)
        self.use_dtw = self.dtw_gamma is not None
        self.regularizer_losses = []

        self.window_len_left = 50
        self.window_len_right = 50

        if self.n_units_encoder is None:
            self.units_encoder = [self.k] * (self.n_layers_encoder - 1)
        elif isinstance(self.n_units_encoder, str):
            self.units_encoder = [int(x) for x in self.n_units_encoder.split()]
            if len(self.units_encoder) == 1:
                self.units_encoder = [self.units_encoder[0]] * (self.n_layers_encoder - 1)
        elif isinstance(self.n_units_encoder, int):
            self.units_encoder = [self.n_units_encoder] * (self.n_layers_encoder - 1)
        else:
            self.units_encoder = self.n_units_encoder
        assert len(self.units_encoder) == (self.n_layers_encoder - 1), 'Misalignment in number of layers between n_layers_encoder and n_units_encoder.'

        if self.n_units_decoder is None:
            self.units_decoder = [self.k] * (self.n_layers_decoder - 1)
        elif isinstance(self.n_units_decoder, str):
            self.units_decoder = [int(x) for x in self.n_units_decoder.split()]
            if len(self.units_decoder) == 1:
                self.units_decoder = [self.units_decoder[0]] * (self.n_layers_decoder - 1)
        elif isinstance(self.n_units_decoder, int):
            self.units_decoder = [self.n_units_decoder] * (self.n_layers_decoder - 1)
        else:
            self.units_decoder = self.n_units_decoder
        assert len(self.units_decoder) == (self.n_layers_decoder - 1), 'Misalignment in number of layers between n_layers_decoder and n_units_decoder.'

        if self.segment_encoding_correspondence_regularizer_scale and \
                self.encoder_type.lower() in ['cnn_hmlstm' ,'hmlstm'] and \
                self.n_layers_encoder == self.n_layers_decoder and \
                self.units_encoder == self.units_decoder[::-1]:
            self.regularize_correspondences = True
        else:
            self.regularize_correspondences = False

        if self.regularize_correspondences and self.resample_inputs == self.resample_outputs:
            self.matched_correspondences = True
        else:
            self.matched_correspondences = False

        if self.pad_seqs:
            if self.mask_padding and ('hmlstm' in self.encoder_type.lower() or 'rnn' in self.encoder_type.lower()):
                self.input_padding = 'post'
            else:
                self.input_padding = 'pre'
            self.target_padding = 'post'
        else:
            self.input_padding = None
            self.target_padding = None

        if self.resample_inputs:
            if self.max_len < self.resample_inputs:
                self.n_timestamps_input = self.max_len
            else:
                self.n_timestamps_input = self.resample_inputs
        else:
            self.n_timestamps_input = self.max_len

        if self.resample_outputs:
            if self.max_len < self.resample_outputs:
                self.n_timesteps_output = self.max_len
            else:
                self.n_timesteps_output = self.resample_outputs
        else:
            self.n_timesteps_output = self.max_len

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.entropy_regularizer_scale:
                    self.entropy_regularizer = binary_entropy_regularizer(
                        scale=self.entropy_regularizer_scale,
                        from_logits=False,
                        session=self.sess
                    )
                else:
                    self.entropy_regularizer = None

                if self.boundary_regularizer_scale:
                    self.boundary_regularizer = lambda bit_probs: tf.reduce_mean(bit_probs) * self.boundary_regularizer_scale
                else:
                    self.boundary_regularizer = None

                if self.segment_encoding_correspondence_regularizer_scale:
                    # self.segment_encoding_correspondence_regularizer = mse_regularizer(scale=self.segment_encoding_correspondence_regularizer_scale, session=self.sess)
                    # self.segment_encoding_correspondence_regularizer = cross_entropy_regularizer(scale=self.segment_encoding_correspondence_regularizer_scale, session=self.sess)
                    self.segment_encoding_correspondence_regularizer = tf.contrib.layers.l1_regularizer(scale=self.segment_encoding_correspondence_regularizer_scale)
                else:
                    self.segment_encoding_correspondence_regularizer = None

        self.label_map = None
        if self.label_map_file:
            if os.path.exists(self.label_map_file):
                self.label_map = pd.read_csv(self.label_map_file)
            else:
                sys.stderr.write('Label map file %s does not exist. Label mapping will not be used.' %self.label_map_file)

    def _pack_metadata(self):
        if hasattr(self, 'n_train'):
            n_train = self.n_train
        else:
            n_train = None
        md = {
            'k': self.k,
            'speaker_list': self.speaker_list,
            'n_train': n_train
        }
        for kwarg in AcousticEncoderDecoder._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.k = md.pop('k')
        self.speaker_list = md.pop('speaker_list', [])
        self.n_train = md.pop('n_train', None)
        for kwarg in AcousticEncoderDecoder._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    def __getstate__(self):
        return self._pack_metadata()

    def __setstate__(self, state):
        self._unpack_metadata(state)
        self._initialize_session()
        self._initialize_metadata()






    ############################################################
    # Private model construction methods
    ############################################################

    def build(self, n_train=None, outdir=None, restore=True, verbose=True):
        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './dnnseg_model/'
        else:
            self.outdir = outdir

        if n_train is not None:
            self.n_train = n_train

        if not hasattr(self, 'n_train') or self.n_train is None:
            raise ValueError('Build parameter "n_train" must be provided the first time build() is called.')

        self._initialize_inputs()
        if self.task != 'streaming_autoencoder':
            self.encoder = self._initialize_encoder(self.X)
            self.encoding = self._initialize_classifier(self.encoder)
            self.decoder_in, self.extra_dims = self._augment_encoding(self.encoding, encoder=self.encoder)
            self.decoder = self._initialize_decoder(self.decoder_in, self.n_timesteps_output)
            self._initialize_output_model()
        self._initialize_objective(n_train)
        self._initialize_ema()
        self._initialize_saver()
        self._initialize_logging()

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )
        self.load(restore=restore)

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                X_n_feats = self.n_coef * (self.order + 1)
                if self.speaker_emb_dim:
                    self.speaker_table, self.speaker_embedding_matrix = initialize_embeddings(
                        self.speaker_list,
                        self.speaker_emb_dim,
                        session=self.sess
                    )
                    self.speaker = tf.placeholder(tf.string, shape=[None], name='speaker')
                    self.speaker_embeddings = tf.nn.embedding_lookup(
                        self.speaker_embedding_matrix,
                        self.speaker_table.lookup(self.speaker)
                    )

                self.X_n_feats = X_n_feats

                self.X = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps_input, X_n_feats), name='X')
                self.X_mask = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps_input), name='X_mask')

                self.X_feat_mean = tf.reduce_sum(self.X, axis=-2) / tf.reduce_sum(self.X_mask, axis=-1, keepdims=True)
                self.X_time_mean = tf.reduce_mean(self.X, axis=-1)

                if self.speaker_emb_dim:
                    tiled_embeddings = tf.tile(self.speaker_embeddings[:, None, :], [1, tf.shape(self.X)[1], 1])
                    self.inputs = tf.concat([self.X, tiled_embeddings], axis=-1)
                else:
                    self.inputs = self.X

                if self.reconstruct_deltas:
                    self.frame_dim = self.n_coef * (self.order + 1)
                else:
                    self.frame_dim = self.n_coef
                self.y = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps_output, self.frame_dim), name='y')

                self.y_mask = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps_output), name='y_mask')

                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_step'
                )
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_batch_step'
                )
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)
                self.batch_len = tf.shape(self.inputs)[0]

                self.loss_summary = tf.placeholder(tf.float32, name='loss_summary')
                self.homogeneity = tf.placeholder(tf.float32, name='homogeneity')
                self.completeness = tf.placeholder(tf.float32, name='completeness')
                self.v_measure = tf.placeholder(tf.float32, name='v_measure')
                self.ami = tf.placeholder(tf.float32, name='ami')
                self.fmi = tf.placeholder(tf.float32, name='fmi')

                self.training = tf.placeholder(tf.bool, name='training')

                if 'hmlstm' in self.encoder_type.lower():
                    self.segmentation_scores = []
                    for i in range(self.n_layers_encoder - 1):
                        self.segmentation_scores.append({'phn': None, 'wrd': None})
                        for s in ['phn', 'wrd']:
                            self.segmentation_scores[-1][s] = {
                                'b_p': tf.placeholder(dtype=self.FLOAT_TF, shape=[], name='b_p_%d_%s' %(i+1, s)),
                                'b_r': tf.placeholder(dtype=self.FLOAT_TF, shape=[], name='b_r_%d_%s' %(i+1, s)),
                                'b_f': tf.placeholder(dtype=self.FLOAT_TF, shape=[], name='b_f_%d_%s' %(i+1, s)),
                                'w_p': tf.placeholder(dtype=self.FLOAT_TF, shape=[], name='w_p_%d_%s' %(i+1, s)),
                                'w_r': tf.placeholder(dtype=self.FLOAT_TF, shape=[], name='w_r_%d_%s' %(i+1, s)),
                                'w_f': tf.placeholder(dtype=self.FLOAT_TF, shape=[], name='w_f_%d_%s' %(i+1, s))
                            }

                if self.task == 'streaming_autoencoder':
                    self.new_series = tf.placeholder(self.FLOAT_TF)

                if self.n_correspondence:
                    self.correspondence_embedding_placeholders = []
                    self.correspondence_feature_placeholders = []
                    self.correspondence_features = []
                    for l in range(self.n_layers_encoder - 1):
                        correspondence_embedding = tf.Variable(
                            tf.zeros(shape=[self.n_correspondence, self.units_encoder[l]]),
                            dtype=self.FLOAT_TF,
                            trainable=False,
                            name='embedding_correspondence_l%d' %l
                        )
                        correspondence_features = tf.Variable(
                            tf.zeros(shape=[self.n_correspondence, self.resample_correspondence, self.frame_dim]),
                            dtype=self.FLOAT_TF,
                            trainable=False,
                            name='X_correspondence_l%d' %l
                        )

                        self.correspondence_embedding_placeholders.append(correspondence_embedding)
                        self.correspondence_feature_placeholders.append(correspondence_features)
                        
                        # TODO: Implement speaker embeddings in the correspondence loss

    def _initialize_encoder(self, encoder_in):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.batch_normalize_encodings:
                    encoding_batch_normalization_decay = self.encoder_batch_normalization_decay
                else:
                    encoding_batch_normalization_decay = None

                if self.mask_padding:
                    mask = self.X_mask
                else:
                    mask = None

                encoder = encoder_in
                if self.input_dropout_rate is not None:
                    encoder = tf.layers.dropout(
                        encoder,
                        self.input_dropout_rate,
                        noise_shape=[tf.shape(encoder)[0], tf.shape(encoder)[1], 1],
                        training=self.training
                    )

                units_utt = self.k
                if self.emb_dim:
                    units_utt += self.emb_dim

                if self.encoder_type.lower() in ['cnn_hmlstm', 'hmlstm']:
                    if self.encoder_type == 'cnn_hmlstm':
                        encoder = Conv1DLayer(
                            self.conv_kernel_size,
                            training=self.training,
                            n_filters=self.n_coef * (self.order + 1),
                            padding='same',
                            activation=tf.nn.elu,
                            batch_normalization_decay=self.encoder_batch_normalization_decay,
                            session=self.sess
                        )(encoder)

                    self.segmenter = HMLSTMSegmenter(
                        self.units_encoder + [units_utt],
                        self.n_layers_encoder,
                        training=self.training,
                        activation=self.encoder_inner_activation,
                        inner_activation=self.encoder_inner_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        boundary_activation=self.encoder_boundary_activation,
                        bottomup_regularizer=self.encoder_weight_regularization,
                        recurrent_regularizer=self.encoder_weight_regularization,
                        topdown_regularizer=self.encoder_weight_regularization,
                        boundary_regularizer=self.encoder_weight_regularization,
                        bias_regularizer=None,
                        layer_normalization=self.encoder_layer_normalization,
                        refeed_boundary=True,
                        power=self.boundary_power,
                        boundary_slope_annealing_rate=self.boundary_slope_annealing_rate,
                        state_slope_annealing_rate=self.state_slope_annealing_rate,
                        slope_annealing_max=self.slope_annealing_max,
                        state_discretizer=self.encoder_state_discretizer,
                        global_step=self.global_step,
                        implementation=2
                    )
                    self.segmenter_output = self.segmenter(encoder, mask=mask)

                    self.regularizer_losses += self.segmenter.get_regularizer_losses()
                    self.segmentation_probs = self.segmenter_output.boundary(discrete=False, as_logits=False)
                    self.encoder_hidden_states = self.segmenter_output.state(discrete=False)


                    for i, l in enumerate(self.segmentation_probs):
                        self._regularize(l, self.entropy_regularizer)
                        self._regularize(l, self.boundary_regularizer)
                        if self.lm_scale:
                            lm = RNNLayer(
                                training=self.training,
                                units=self.units_encoder[i],
                                activation=self.encoder_inner_activation,
                                recurrent_activation=self.encoder_recurrent_activation,
                                return_sequences=True,
                                session=self.sess
                            )(self.encoder_hidden_states[i][:,:-1], mask=l[:,:-1])
                            if self.encoder_state_discretizer:
                                lm_out = tf.sigmoid(lm)
                                loss_fn = tf.losses.sigmoid_cross_entropy
                            else:
                                lm_out = lm
                                loss_fn = tf.losses.mean_squared_error
                            lm_loss = loss_fn(
                                self.encoder_hidden_states[i][:, 1:],
                                lm_out,
                                weights=l[:, 1:, None]
                            )
                            self._regularize(lm_loss, identity_regularizer(self.lm_scale))

                    encoder = self.segmenter_output.output(return_sequences=self.task == 'streaming_autoencoder')

                    # if encoding_batch_normalization_decay:
                    #     encoder = tf.contrib.layers.batch_norm(
                    #         encoder,
                    #         decay=self.encoder_batch_normalization_decay,
                    #         center=True,
                    #         scale=True,
                    #         zero_debias_moving_mean=True,
                    #         is_training=self.training,
                    #         updates_collections=None
                    #     )

                    encoder = DenseLayer(
                        training=self.training,
                        units=units_utt,
                        activation=self.encoder_activation,
                        batch_normalization_decay=encoding_batch_normalization_decay,
                        session=self.sess
                    )(encoder)

                elif self.encoder_type.lower() in ['rnn', 'cnn_rnn']:
                    if self.encoder_type == 'cnn_rnn':
                        encoder = Conv1DLayer(
                            self.conv_kernel_size,
                            training=self.training,
                            n_filters=self.n_coef * (self.order + 1),
                            padding='same',
                            activation=tf.nn.elu,
                            batch_normalization_decay=self.encoder_batch_normalization_decay,
                            session=self.sess
                        )(encoder)

                    encoder = MultiRNNLayer(
                        training=self.training,
                        units=self.units_encoder + [units_utt],
                        layers=self.n_layers_encoder,
                        activation=self.encoder_inner_activation,
                        inner_activation=self.encoder_inner_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        return_sequences=False,
                        name='RNNEncoder',
                        session=self.sess
                    )(encoder, mask=mask)

                    # encoder = DenseLayer(
                    #     training=self.training,
                    #     units=units_utt,
                    #     activation=self.encoder_activation,
                    #     batch_normalization_decay=encoding_batch_normalization_decay,
                    #     session=self.sess
                    # )(encoder)

                    if encoding_batch_normalization_decay:
                        encoder = tf.contrib.layers.batch_norm(
                            encoder,
                            decay=self.encoder_batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None
                        )


                elif self.encoder_type.lower() == 'cnn':
                    encoder = Conv1DLayer(
                        self.conv_kernel_size,
                        training=self.training,
                        n_filters=self.n_coef * (self.order + 1),
                        padding='same',
                        activation=tf.nn.elu,
                        batch_normalization_decay=self.encoder_batch_normalization_decay,
                        session=self.sess
                    )(encoder)

                    for i in range(self.n_layers_encoder - 1):
                        if i > 0 and self.encoder_resnet_n_layers_inner:
                            encoder = Conv1DResidualLayer(
                                self.conv_kernel_size,
                                training=self.training,
                                n_filters=self.units_encoder[i],
                                padding='causal',
                                layers_inner=self.encoder_resnet_n_layers_inner,
                                activation=self.encoder_inner_activation,
                                activation_inner=self.encoder_inner_activation,
                                batch_normalization_decay=self.batch_normalization_decay,
                                session=self.sess
                            )(encoder)
                        else:
                            encoder = Conv1DLayer(
                                self.conv_kernel_size,
                                training=self.training,
                                n_filters=self.units_encoder[i],
                                padding='causal',
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)

                    encoder = DenseLayer(
                        training=self.training,
                        units=units_utt,
                        activation=self.encoder_activation,
                        batch_normalization_decay=encoding_batch_normalization_decay,
                        session=self.sess
                    )(tf.layers.Flatten()(encoder))

                elif self.encoder_type.lower() == 'dense':
                    encoder = tf.layers.Flatten()(encoder)

                    for i in range(self.n_layers_encoder - 1):
                        if i > 0 and self.encoder_resnet_n_layers_inner:
                            encoder = DenseResidualLayer(
                                training=self.training,
                                units=self.n_timesteps_input * self.units_encoder[i],
                                layers_inner=self.encoder_resnet_n_layers_inner,
                                activation=self.encoder_inner_activation,
                                activation_inner=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)
                        else:
                            encoder = DenseLayer(
                                training=self.training,
                                units=self.n_timesteps_input * self.units_encoder[i],
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)

                    encoder = DenseLayer(
                        training=self.training,
                        units=units_utt,
                        activation=self.encoder_activation,
                        batch_normalization_decay=encoding_batch_normalization_decay,
                        session=self.sess
                    )(encoder)

                else:
                    raise ValueError('Encoder type "%s" is not currently supported' %self.encoder_type)

                return encoder

    def _initialize_classifier(self, classifier_in):
        self.encoding = None
        raise NotImplementedError

    def _augment_encoding(self, encoding, encoder=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.task == 'utterance_classifier':
                    if self.binary_classifier:
                        self.labels = binary2integer(tf.round(encoding), session=self.sess)
                        self.label_probs = bernoulli2categorical(encoding, session=self.sess)
                    else:
                        self.labels = tf.argmax(self.encoding, axis=-1)
                        self.label_probs = self.encoding

                extra_dims = None

                if encoder is not None:
                    if self.emb_dim:
                        extra_dims = tf.nn.elu(encoder[:,self.k:])

                    if self.decoder_use_input_length or self.utt_len_emb_dim:
                        utt_len = tf.reduce_sum(self.y_mask, axis=1, keepdims=True)
                        if self.decoder_use_input_length:
                            if extra_dims is None:
                                extra_dims = utt_len
                            else:
                                extra_dims = tf.concat(
                                [extra_dims, utt_len],
                                axis=1
                            )

                        if self.utt_len_emb_dim:
                            self.utt_len_emb_mat = tf.identity(
                                tf.Variable(
                                    tf.random_uniform([int(self.y_mask.shape[1]) + 1, self.utt_len_emb_dim], -1., 1.),
                                    dtype=self.FLOAT_TF
                                )
                            )

                            if self.optim_name == 'Nadam':
                                # Nadam breaks with sparse gradients, have to use matmul
                                utt_len_emb = tf.one_hot(tf.cast(utt_len[:, 0], dtype=self.INT_TF), int(self.y_mask.shape[1]) + 1)
                                utt_len_emb = tf.matmul(utt_len_emb, self.utt_len_emb_mat)
                            else:
                                utt_len_emb = tf.gather(self.utt_len_emb_mat, tf.cast(utt_len[:, 0], dtype=self.INT_TF), axis=0)

                            if extra_dims is None:
                                extra_dims = utt_len_emb
                            else:
                                extra_dims = tf.concat(
                                    [extra_dims,
                                     utt_len_emb],
                                    axis=1
                                )

                if self.speaker_emb_dim:
                    speaker_embeddings = self.speaker_embeddings
                    if extra_dims is None:
                        extra_dims = speaker_embeddings
                    else:
                        extra_dims = tf.concat([extra_dims, speaker_embeddings], axis=1)

                if extra_dims is not None:
                    decoder_in = tf.concat([encoding, extra_dims], axis=1)
                else:
                    decoder_in = encoding

                return decoder_in, extra_dims

    def _initialize_decoder(self, decoder_in, n_timesteps):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                decoder = decoder_in
                if self.mask_padding:
                    mask = self.y_mask
                else:
                    mask = None

                if self.decoder_type.lower() == 'rnn':
                    tile_dims = [1] * (len(decoder.shape) + 1)
                    tile_dims[-2] = n_timesteps

                    decoder = tf.tile(
                        decoder[..., None, :],
                        tile_dims
                    )
                    decoder *= self.y_mask[...,None]
                    max_len = tf.reduce_sum(self.y_mask, axis=1, keepdims=True)[..., None]

                    index = tf.range(self.y.shape[1])[None, ..., None]
                    index = tf.tile(
                        index,
                        [self.batch_len, 1, 1]
                    )
                    index = tf.cast(index, dtype=self.FLOAT_TF)
                    index /= max_len

                    decoder = tf.concat([decoder, index], axis=2)

                    decoder = MultiRNNLayer(
                        training=self.training,
                        units=self.units_decoder + [self.frame_dim],
                        layers=self.n_layers_decoder,
                        activation=self.decoder_inner_activation,
                        inner_activation=self.decoder_inner_activation,
                        recurrent_activation=self.decoder_recurrent_activation,
                        return_sequences=True,
                        name='RNNDecoder',
                        session=self.sess
                    )(decoder, mask=mask)

                    # if self.decoder_activation is not None:
                    #     s = tf.Variable(tf.ones([], dtype=self.FLOAT_TF), name='output_scale')
                    #     s = tf.Print(s, [s])
                    #     decoder *= s

                    decoder = DenseLayer(
                        training=self.training,
                        units=self.frame_dim,
                        activation=self.decoder_activation,
                        batch_normalization_decay=self.decoder_batch_normalization_decay,
                        session=self.sess
                    )(decoder)

                elif self.decoder_type.lower() == 'cnn':
                    assert n_timesteps is not None, 'n_timesteps must be defined when decoder_type == "cnn"'

                    decoder = DenseLayer(
                        training=self.training,
                        units=n_timesteps * self.units_decoder[0],
                        activation=tf.nn.elu,
                        batch_normalization_decay=self.decoder_batch_normalization_decay,
                        session=self.sess
                    )(decoder)

                    decoder_shape = tf.concat([tf.shape(decoder)[:-2], [n_timesteps, self.units_decoder[0]]], axis=0)
                    decoder = tf.reshape(decoder, decoder_shape)

                    for i in range(self.n_layers_decoder - 1):
                        if i > 0 and self.decoder_resnet_n_layers_inner:
                            decoder = Conv1DResidualLayer(
                                self.conv_kernel_size,
                                training=self.training,
                                n_filters=self.units_decoder[i],
                                padding='same',
                                layers_inner=self.decoder_resnet_n_layers_inner,
                                activation=self.decoder_inner_activation,
                                activation_inner=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)
                        else:
                            decoder = Conv1DLayer(
                                self.conv_kernel_size,
                                training=self.training,
                                n_filters=self.units_decoder[i],
                                padding='same',
                                activation=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)

                        self._regularize_correspondences(self.n_layers_encoder - i - 2, decoder)

                    decoder = DenseLayer(
                            training=self.training,
                            units=n_timesteps * self.frame_dim,
                            activation=self.decoder_inner_activation,
                            batch_normalization_decay=False,
                            session=self.sess
                    )(tf.layers.Flatten()(decoder))

                    decoder_shape = tf.concat([tf.shape(decoder)[:-2], [n_timesteps, self.frame_dim]], axis=0)
                    decoder = tf.reshape(decoder, decoder_shape)

                elif self.decoder_type.lower() == 'dense':
                    assert n_timesteps is not None, 'n_timesteps must be defined when decoder_type == "dense"'

                    for i in range(self.n_layers_decoder - 1):

                        in_shape_flattened, out_shape_unflattened = self._get_decoder_shapes(decoder, n_timesteps, self.units_decoder[i], expand_sequence=i==0)
                        decoder = tf.reshape(decoder, in_shape_flattened)

                        if i > 0 and self.decoder_resnet_n_layers_inner:
                            if self.units_decoder[i] != self.units_decoder[i-1]:
                                project_inputs = True
                            else:
                                project_inputs = False

                            decoder = DenseResidualLayer(
                                training=self.training,
                                units=n_timesteps * self.units_decoder[i],
                                layers_inner=self.decoder_resnet_n_layers_inner,
                                activation=self.decoder_inner_activation,
                                activation_inner=self.decoder_inner_activation,
                                project_inputs=project_inputs,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)
                        else:
                            decoder = DenseLayer(
                                training=self.training,
                                units=n_timesteps * self.units_decoder[i],
                                activation=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)

                        decoder = tf.reshape(decoder, out_shape_unflattened)

                        self._regularize_correspondences(self.n_layers_encoder - i - 2, decoder)

                    in_shape_flattened, out_shape_unflattened = self._get_decoder_shapes(decoder, n_timesteps, self.frame_dim)
                    decoder = tf.reshape(decoder, in_shape_flattened)

                    decoder = DenseLayer(
                        training=self.training,
                        units=n_timesteps * self.frame_dim,
                        activation=self.decoder_activation,
                        batch_normalization_decay=None,
                        session=self.sess
                    )(decoder)

                    decoder = tf.reshape(decoder, out_shape_unflattened)

                elif self.decoder_type.lower() == 'layerwise_rnn':
                    assert self.n_layers_encoder == self.n_layers_decoder, 'layerwise_rnn requires equal number of layers in encoder and decoder.'
                    assert self.units_encoder == self.units_decoder[::-1], 'layerwise_rnn requires equal units in each encoder and decoder layer.'
                    self.decoder_layers = []

                    for i in range(self.n_layers_decoder):
                        if i == 0:
                            tile_dims = [1] * len(decoder.shape) + 1
                            tile_dims[-2] = tf.shape(self.y)[1]
                            input_cur = tf.tile(
                                decoder[..., None, :],
                                tile_dims
                            ) * self.y_mask[..., None]
                        else:
                            input_cur = self.encoder_hidden_states[self.n_layers_decoder - i - 1]

                        output_cur = RNNLayer(
                            training=self.training,
                            units=self.units_decoder[i] if i < (self.n_layers_decoder - 1) else self.frame_dim,
                            activation=self.decoder_inner_activation,
                            recurrent_activation=self.decoder_recurrent_activation,
                            return_sequences=True,
                            session=self.sess
                        )(input_cur)

                        if i == self.n_layers_decoder - 1:
                            output_cur = DenseLayer(
                                training=self.training,
                                units=self.frame_dim,
                                activation=self.decoder_activation,
                                batch_normalization_decay=None,
                                session=self.sess
                            )(output_cur)

                        self.decoder_layers.append(output_cur)

                    decoder = self.decoder_layers[-1]

                elif self.decoder_type.lower() == 'layerwise_dense':
                    assert n_timesteps is not None, 'n_timesteps must be defined when decoder_type == "dense"'
                    assert self.n_layers_encoder == self.n_layers_decoder, 'layerwise_dense requires equal number of layers in encoder and decoder.'
                    assert self.units_encoder == self.units_decoder[::-1], 'layerwise_dense requires equal units in each encoder and decoder layer.'
                    self.decoder_layers = []

                    for i in range(self.n_layers_decoder - 1):
                        if i == 0:
                            input_cur = decoder
                        else:
                            input_cur = self.encoder_hidden_states[self.n_layers_decoder - i - 1]

                        input_cur = tf.layers.Flatten()(input_cur)

                        if i > 0 and self.decoder_resnet_n_layers_inner:
                            if self.units_decoder[i] != self.units_decoder[i-1]:
                                project_inputs = True
                            else:
                                project_inputs = False

                            decoder_layer = DenseResidualLayer(
                                training=self.training,
                                units=n_timesteps * self.units_decoder[i],
                                layers_inner=self.decoder_resnet_n_layers_inner,
                                activation=self.decoder_inner_activation,
                                activation_inner=self.decoder_inner_activation,
                                project_inputs=project_inputs,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(input_cur)
                        else:
                            decoder_layer = DenseLayer(
                                training=self.training,
                                units=n_timesteps * self.units_decoder[i],
                                activation=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(input_cur)

                        decoder_layer = tf.reshape(decoder_layer, (self.batch_len, n_timesteps, self.units_decoder[i]))
                        self.decoder_layers.append(decoder_layer)

                    input_cur = tf.layers.Flatten()(self.encoder_hidden_states[0])
                    decoder_layer = DenseLayer(
                        training=self.training,
                        units=n_timesteps * self.frame_dim,
                        activation=self.decoder_activation,
                        batch_normalization_decay=None,
                        session=self.sess
                    )(input_cur)

                    decoder_layer = tf.reshape(decoder_layer, (self.batch_len, n_timesteps, self.frame_dim))
                    self.decoder_layers.append(decoder_layer)

                    decoder = self.decoder_layers[-1]

                else:
                    raise ValueError('Decoder type "%s" is not currently supported' %self.decoder_type)

                return decoder

    def _initialize_output_model(self):
        self.out = None
        raise NotImplementedError

    def _initialize_objective(self, n_train):
        self.reconst = None
        self.encoding_post = None
        self.labels = None
        self.labels_post = None
        self.label_probs = None
        self.label_probs_post = None
        raise NotImplementedError

    def _initialize_optimizer(self, name):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
                if name is None:
                    self.lr = lr
                    return None
                if self.lr_decay_family is not None:
                    lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
                    lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
                    lr_decay_staircase = self.lr_decay_staircase
                    self.lr = getattr(tf.train, self.lr_decay_family)(
                        lr,
                        self.global_step,
                        lr_decay_steps,
                        lr_decay_rate,
                        staircase=lr_decay_staircase,
                        name='learning_rate'
                    )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(np.inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                clip = self.max_global_gradient_norm

                return {
                    'SGD': lambda x: self._clipped_optimizer_class(tf.train.GradientDescentOptimizer)(x, max_global_norm=clip) if clip else tf.train.GradientDescentOptimizer(x),
                    'Momentum': lambda x: self._clipped_optimizer_class(tf.train.MomentumOptimizer)(x, 0.9, max_global_norm=clip) if clip else tf.train.MomentumOptimizer(x, 0.9),
                    'AdaGrad': lambda x: self._clipped_optimizer_class(tf.train.AdagradOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: self._clipped_optimizer_class(tf.train.AdadeltaOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: self._clipped_optimizer_class(tf.train.AdamOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdamOptimizer(x),
                    'FTRL': lambda x: self._clipped_optimizer_class(tf.train.FtrlOptimizer)(x, max_global_norm=clip) if clip else tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: self._clipped_optimizer_class(tf.train.RMSPropOptimizer)(x, max_global_norm=clip) if clip else tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: self._clipped_optimizer_class(tf.contrib.opt.NadamOptimizer)(x, max_global_norm=clip) if clip else tf.contrib.opt.NadamOptimizer(x)
                }[name](self.lr)

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                tf.summary.scalar('loss_summary', self.loss_summary, collections=['metrics'])
                if self.task == 'utterance_classifier':
                    tf.summary.scalar('homogeneity', self.homogeneity, collections=['metrics'])
                    tf.summary.scalar('completeness', self.completeness, collections=['metrics'])
                    tf.summary.scalar('v_measure', self.v_measure, collections=['metrics'])
                    tf.summary.scalar('ami', self.ami, collections=['metrics'])
                    tf.summary.scalar('fmi', self.fmi, collections=['metrics'])

                elif self.task == 'streaming_autoencoder':
                    pass

                else:
                    if 'hmlstm' in self.encoder_type.lower():
                        for i in range(self.n_layers_encoder - 1):
                            for s in ['phn', 'wrd']:
                                tf.summary.scalar('b_p_%d_%s' %(i+1, s), self.segmentation_scores[i][s]['b_p'], collections=['segmentations'])
                                tf.summary.scalar('b_r_%d_%s' %(i+1, s), self.segmentation_scores[i][s]['b_r'], collections=['segmentations'])
                                tf.summary.scalar('b_f_%d_%s' %(i+1, s), self.segmentation_scores[i][s]['b_f'], collections=['segmentations'])
                                tf.summary.scalar('w_p_%d_%s' % (i+1, s), self.segmentation_scores[i][s]['b_p'], collections=['segmentations'])
                                tf.summary.scalar('w_r_%d_%s' % (i+1, s), self.segmentation_scores[i][s]['b_r'], collections=['segmentations'])
                                tf.summary.scalar('w_f_%d_%s' % (i+1, s), self.segmentation_scores[i][s]['b_f'], collections=['segmentations'])

                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dnnseg', self.sess.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dnnseg')
                self.summary_metrics = tf.summary.merge_all(key='metrics')
                self.summary_segmentations = tf.summary.merge_all(key='segmentations')

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

                self.check_numerics_ops = [tf.check_numerics(v, 'Numerics check failed') for v in tf.trainable_variables()]

    def _initialize_ema(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.ema_decay:
                    vars = [var for var in tf.get_collection('trainable_variables') if 'BatchNorm' not in var.name]

                    self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
                    self.ema_op = self.ema.apply(vars)
                    self.ema_map = {}
                    for v in vars:
                        self.ema_map[self.ema.average_name(v)] = v
                    self.ema_saver = tf.train.Saver(self.ema_map)




    ############################################################
    # Private Soft-DTW methods
    ############################################################

    def _pairwise_distances(self, targets, preds):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                targets = tf.expand_dims(targets, axis=-2)
                preds = tf.expand_dims(preds, axis=-3)
                offsets = targets - preds

                distances = tf.norm(offsets, axis=-1)

                return distances

    def _min_smoothed(self, input, gamma, axis=-1, keepdims=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = tf.convert_to_tensor(input, dtype=self.FLOAT_TF)
                out = -out / gamma
                out = -gamma * tf.reduce_logsumexp(out, axis=axis, keepdims=keepdims)
                return out

    def _dtw_compute_cell(self, D_ij, R_im1_jm1, R_im1_j, R_i_jm1, gamma):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                r_ij = D_ij + self._min_smoothed([R_im1_jm1, R_im1_j, R_i_jm1], gamma, axis=0)

                return r_ij

    def _dtw_inner_scan(self, D_i, R_im1, R_i0, gamma):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Compute dimensions
                b = tf.shape(D_i)[1:]

                # Extract alignment scores from adjacent cells in preceding row, prepending upper-left alignment to R_im1_jm1
                R_im1_jm1 = tf.concat(
                    [
                        tf.fill(tf.concat([[1],b], axis=0), R_i0),
                        R_im1[:-1, :]
                    ],
                    axis=0
                )
                R_im1_j = R_im1

                # Scan over columns of D (prediction indices)
                out = tf.scan(
                    lambda a, x: self._dtw_compute_cell(x[0], x[1], x[2], a, gamma),
                    [D_i, R_im1_jm1, R_im1_j],
                    initializer=tf.fill(b, np.inf),
                    swap_memory=True
                )

                return out

    def _dtw_outer_scan(self, D, gamma):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Extract dimensions
                n = tf.shape(D)[0]
                m = tf.shape(D)[1]
                b = tf.shape(D)[2:]

                # Construct the 0th column with appropriate dimensionality
                R_i0 = tf.concat([[0.], tf.fill([n-1], np.inf)], axis=0)

                # Scan over rows of D (target indices)
                out = tf.scan(
                    lambda a, x: self._dtw_inner_scan(x[0], a, x[1], gamma),
                    [D, R_i0],
                    initializer=tf.fill(tf.concat([[m], b], axis=0), np.inf),
                    swap_memory=True
                )

                return out

    def _soft_dtw_A(self, targets, preds, gamma, targets_mask=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if targets_mask is None:
                    targets_mask = tf.ones(tf.shape(targets)[:-1])

                targets_mask = tf.cast(targets_mask, tf.bool)

                D = self._pairwise_distances(targets, preds)
                n = int(D.shape[-2])
                m = int(D.shape[-1])

                R = {(0,0): tf.fill([tf.shape(D)[0]], 0.)}

                inf = tf.fill([tf.shape(D)[0]], np.inf)
                zero = tf.zeros([tf.shape(D)[0]])

                for i in range(1, n + 1):
                    R[(i,0)] = inf
                for j in range(1, m + 1):
                    R[(0,j)] = inf

                for j in range(1, m+1):
                    for i in range(1, n+1):
                        r_ij = D[:,i-1,j-1] + self._min_smoothed(tf.stack([R[(i-1,j-1)], R[(i-1,j)], R[(i, j-1)]], axis=1), gamma)
                        R[(i,j)] = tf.where(
                            targets_mask[:,i-1],
                            r_ij,
                            zero
                        )

                out = tf.reduce_mean(R[(n,m)])

                return out

    def _soft_dtw_B(self, targets, preds, gamma, mask=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n = tf.shape(targets)[-2]
                b = tf.shape(targets)[-1]

                # Compute/transform mask as needed
                if mask is None:
                    mask = tf.ones([n, b])
                else:
                    mask = tf.transpose(mask, perm=[1, 0])

                # Compute distance matrix
                D = self._pairwise_distances(targets, preds)

                # Move batch dimension(s) to end so we can scan along time dimensions
                perm = list(range(len(D.shape)))
                perm = perm[-2:] + perm[:-2]
                D = tf.transpose(D, perm=perm)

                # Perform soft-DTW alignment
                R = self._dtw_outer_scan(D, gamma)

                # Move batch dimension(s) back to beginning so indexing works as expected
                perm = list(range(len(D.shape)))
                perm = perm[2:] + perm[:2]
                R = tf.transpose(R, perm=perm)

                # Extract final cell of alignment matrix
                # if self.pad_seqs:
                #     target_end_ix = tf.cast(
                #         tf.maximum(
                #             tf.reduce_sum(mask, axis=0) - 1,
                #             tf.zeros([b], dtype=self.FLOAT_TF)
                #         ),
                #         self.INT_TF
                #     )
                #     out = tf.gather(R[..., -1], target_end_ix, axis=1)
                # else:
                out = R[..., -1, -1]

                return out

    # Numpy sanity checks for Soft-DTW implementation

    def _logsumexp_NP(self, input, axis=-1):
        max_in = np.max(input, axis=axis)
        ds = input - max_in[..., None]
        sum_of_exp = np.exp(ds).sum(axis=axis)
        return max_in + np.log(sum_of_exp)

    def _min_smoothed_NP(self, input, gamma, axis=-1, keepdims=False):
        out = -input / gamma
        out = -gamma * self._logsumexp_NP(out, axis=axis)
        return out

    def _pairwise_distances_NP(self, targets, preds):
        targets = np.expand_dims(targets, axis=2)
        preds = np.expand_dims(preds, axis=1)
        distances = targets - preds

        out = np.linalg.norm(distances, axis=3)

        return out

    def _soft_dtw_NP(self, targets, preds, gamma, targets_mask=None, preds_mask=None):
        if targets_mask is None:
            targets_mask = np.ones(targets.shape[:-1])
        if preds_mask is None:
            preds_mask = np.ones(preds.shape[:-1])
        targets_mask = targets_mask.astype(np.bool)
        preds_mask = preds_mask.astype(np.bool)

        D = self._pairwise_distances_NP(targets, preds)
        n = int(D.shape[1])
        m = int(D.shape[2])

        R = np.zeros((D.shape[0], D.shape[1] + 1, D.shape[2] + 1))

        inf = np.full((D.shape[0],), np.inf)
        zero = np.zeros((D.shape[0],))

        R[:, 1:, 0] = inf
        R[:, 0, 1:] = inf

        for j in range(1, m+1):
            for i in range(1, n+1):
                r_ij = D[:,i-1,j-1] + self._min_smoothed_NP(np.stack([R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]], axis=1), gamma, axis=1)
                R[:,i,j] = np.where(
                    np.logical_and(targets_mask[:,i-1], preds_mask[:,j-1]),
                    r_ij,
                    zero
                )

        return R





    ############################################################
    # Private utility methods
    ############################################################

    def _extract_backward_targets(self, n):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                pass

    ## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
    def _clipped_optimizer_class(self, base_optimizer):
        class ClippedOptimizer(base_optimizer):
            def __init__(self, *args, max_global_norm=None, **kwargs):
                super(ClippedOptimizer, self).__init__( *args, **kwargs)
                self.max_global_norm = max_global_norm

            def compute_gradients(self, *args, **kwargs):
                grads_and_vars = super(ClippedOptimizer, self).compute_gradients(*args, **kwargs)
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))
                return grads_and_vars

            def apply_gradients(self, grads_and_vars, **kwargs):
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))

                return super(ClippedOptimizer, self).apply_gradients(grads_and_vars, **kwargs)

        return ClippedOptimizer

    def _regularize(self, var, regularizer):
        if regularizer is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)

    def _regularize_correspondences(self, layer_number, preds):
        if self.regularize_correspondences:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    states = self.encoder_hidden_states[layer_number]
                    if self.reverse_targets:
                        states = states[:, ::-1, :]

                    if self.matched_correspondences:
                        correspondences = states - preds
                        self._regularize(
                            correspondences,
                            tf.contrib.layers.l2_regularizer(self.segment_encoding_correspondence_regularizer_scale)
                        )
                    else:
                        correspondences = self._soft_dtw_B(states, preds, self.dtw_gamma, mask=self.y_mask)
                        self._regularize(
                            correspondences,
                            tf.contrib.layers.l1_regularizer(self.segment_encoding_correspondence_regularizer_scale)
                        )

    def _get_decoder_shapes(self, decoder, n_timesteps, units, expand_sequence=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                decoder_in_shape = tf.shape(decoder)
                if expand_sequence:
                    decoder_in_shape_flattened = decoder_in_shape
                else:
                    feat = int(decoder.shape[-1])
                    decoder_in_shape_flattened = tf.concat([decoder_in_shape[:-2], [n_timesteps * feat]], axis=0)
                decoder_out_shape = tf.concat([decoder_in_shape_flattened[:-1], [n_timesteps, units]], axis=0)

                return decoder_in_shape_flattened, decoder_out_shape

    def _streaming_dynamic_scan_inner(self, targets, prev, encoder_in, left, cur, right):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                prev_loss, state, _, _ = prev
                if self.batch_normalize_encodings:
                    encoding_batch_normalization_decay = self.encoder_batch_normalization_decay
                else:
                    encoding_batch_normalization_decay = None

                units_utt = self.k
                if self.emb_dim:
                    units_utt += self.emb_dim

                log_loss = self.normalize_data and self.constrain_output

                o, h = self.encoder_cell(encoder_in, state)
                encoder = o[-1][0]
                encoder = DenseLayer(
                    self.training,
                    units=units_utt,
                    activation=self.encoder_activation,
                    batch_normalization_decay=encoding_batch_normalization_decay,
                    session=self.sess
                )(encoder)

                encoding = self._initialize_classifier(encoder)
                decoder_in, extra_dims = self._augment_encoding(encoding, encoder=encoder)

                targ_left = targets[cur:left:-1]
                with tf.variable_scope('left'):
                    pred_left = self._initialize_decoder(decoder_in, self.window_len_left)
                with tf.variable_scope('right'):
                    pred_right = self._initialize_decoder(decoder_in, self.window_len_right)

                loss_left = self._get_loss(targ_left, pred_left, log_loss=log_loss)
                loss_left /= self.window_len_left

                targ_right = targets[cur + 1:right + 1]
                loss_right = self._get_loss(targ_right, pred_right, log_loss=log_loss)
                loss_right /= self.window_len_right

                loss = loss_left + loss_right

                return (prev_loss + loss, h, pred_left, pred_right)

    def _streaming_dynamic_scan(self, input, reset_state=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Make input time major
                batch_size = tf.shape(input)[0]
                n_timesteps = tf.shape(input)[1]
                input = tf.transpose(input, [1, 0, 2])

                targets = self.X
                if not self.reconstruct_deltas:
                    targets = targets[..., :self.n_coef]
                targets = tf.pad(targets, [[0, 0], [self.window_len_left, self.window_len_right], [0, 0]])
                targets = tf.transpose(targets, [1, 0, 2])
                t_lb = tf.range(0, tf.shape(self.X)[1])
                t = t_lb + self.window_len_left
                t_rb = t + self.window_len_right

                losses, states, preds_left, preds_right = tf.scan(
                    lambda a, x: self._streaming_dynamic_scan_inner(targets, a, x[0], x[1], x[2], x[3]),
                    [input, t_lb, t, t_rb],
                    initializer=(
                        0., # Initial loss
                        self.encoder_state, # Initial state
                        tf.zeros([batch_size, self.window_len_left, self.frame_dim]), # Initial left prediction
                        tf.zeros([batch_size, self.window_len_right, self.frame_dim]) # Initial right prediction
                    )
                )

                return losses[-1], states, preds_left, preds_right

    def _map_state(self, state):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {}
                for i in range(len(self.encoder_state)):
                    for j in range(len(self.encoder_state[i])):
                        fd[self.encoder_state[i][j]] = state[i][j]

                return fd

    def _get_segs_and_states(self, X, X_mask, training_batch_norm=False, training_dropout=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.pad_seqs:
                    if not np.isfinite(self.eval_minibatch_size):
                        minibatch_size = len(X)
                    else:
                        minibatch_size = self.eval_minibatch_size
                else:
                    minibatch_size = 1

                segmentation_probs = []
                states = []

                for i in range(0, len(X), minibatch_size):
                    if self.pad_seqs:
                        indices = slice(i, i + minibatch_size, 1)
                    else:
                        indices = i

                    fd_minibatch = {
                        self.X: X[indices],
                        self.X_mask: X_mask[indices],
                        self.training: training_batch_norm,
                        self.training: training_dropout
                    }

                    segmentation_probs_cur, states_cur = self.sess.run(
                        [self.segmentation_probs, self.encoder_hidden_states[:-1]],
                        feed_dict=fd_minibatch
                    )

                    segmentation_probs.append(np.stack(segmentation_probs_cur, axis=-1))
                    states.append(np.concatenate(states_cur, axis=-1))

                new_segmentation_probs = [np.squeeze(x, axis=-1) for x in np.split(np.concatenate(segmentation_probs, axis=0), 1, axis=-1)]
                new_states = np.split(np.concatenate(states, axis=0), np.cumsum(self.units_encoder[:-1], dtype='int'), axis=-1)

                return new_segmentation_probs, new_states

    def _collect_previous_segments(self, n, data, segtype=None, X=None, X_mask=None, training_batch_norm=False, training_dropout=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if segtype is None:
                    segtype = self.segtype

                if X is None or X_mask is None:
                    sys.stderr.write('Getting input data...\n')
                    X, X_mask = data.inputs(
                        segments=segtype,
                        padding=self.input_padding,
                        normalize=self.normalize_data,
                        center=self.center_data,
                        resample=self.resample_inputs,
                        max_len=self.max_len
                    )

                sys.stderr.write('Collecting boundary and state predictions...\n')
                segmentation_probs, states = self._get_segs_and_states(X, X_mask, training_batch_norm, training_dropout)

                if 'bsn' in self.encoder_boundary_activation.lower():
                    smoothing_algorithm = None
                else:
                    smoothing_algorithm = 'rbf'

                sys.stderr.write('Converting predictions into tables of segments...\n')
                segment_tables = data.get_segment_tables_from_segmenter_states(
                    segmentation_probs,
                    parent_segment_type=segtype,
                    states=states,
                    algorithm=smoothing_algorithm,
                    mask=X_mask,
                    padding=self.input_padding,
                    discretize=False
                )

                sys.stderr.write('Resegmenting input data...\n')
                y = []

                keep_ix = []

                for l in range(len(segment_tables)):
                    y_cur, _ = data.targets(
                        segments=segment_tables[l],
                        padding='post',
                        reverse=self.reverse_targets,
                        normalize=self.normalize_data,
                        center=self.center_data,
                        with_deltas=self.reconstruct_deltas,
                        resample=self.resample_correspondence
                    )
                    keep_ix.append(np.random.randint(0, len(y_cur), (n,)))

                    y.append(y_cur[keep_ix[l]])

                sys.stderr.write('Collecting segment embeddings...\n')
                n_units = self.units_encoder
                embeddings = []
                for l in range(len(segment_tables)):
                    embeddings_cur = []
                    for f in data.fileIDs:
                        embeddings_cur.append(segment_tables[l][f][['d%d' %u for u in range(n_units[l])]].as_matrix())
                    embeddings.append(np.concatenate(embeddings_cur, axis=0)[keep_ix[l]])

                return embeddings, y

    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround
    def _restore_inner(self, path, predict=False, allow_missing=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                try:
                    if predict:
                        self.ema_saver.restore(self.sess, path)
                    else:
                        self.saver.restore(self.sess, path)
                except tf.errors.DataLossError:
                    sys.stderr.write('Read failure during load. Trying from backup...\n')
                    if predict:
                        self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                    else:
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                except tf.errors.NotFoundError as err:  # Model contains variables that are missing in checkpoint, special handling needed
                    if allow_missing:
                        reader = tf.train.NewCheckpointReader(path)
                        saved_shapes = reader.get_variable_to_shape_map()
                        model_var_names = sorted(
                            [(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                        ckpt_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                                 if var.name.split(':')[0] in saved_shapes])

                        model_var_names_set = set([x[1] for x in model_var_names])
                        ckpt_var_names_set = set([x[1] for x in ckpt_var_names])

                        missing_in_ckpt = model_var_names_set - ckpt_var_names_set
                        if len(missing_in_ckpt) > 0:
                            sys.stderr.write(
                                'Checkpoint file lacked the variables below. They will be left at their initializations.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))
                        missing_in_model = ckpt_var_names_set - model_var_names_set
                        if len(missing_in_model) > 0:
                            sys.stderr.write(
                                'Checkpoint file contained the variables below which do not exist in the current model. They will be ignored.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))

                        restore_vars = []
                        name2var = dict(
                            zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

                        with tf.variable_scope('', reuse=True):
                            for var_name, saved_var_name in ckpt_var_names:
                                curr_var = name2var[saved_var_name]
                                var_shape = curr_var.get_shape().as_list()
                                if var_shape == saved_shapes[saved_var_name]:
                                    restore_vars.append(curr_var)

                        if predict:
                            self.ema_map = {}
                            for v in restore_vars:
                                self.ema_map[self.ema.average_name(v)] = v
                            saver_tmp = tf.train.Saver(self.ema_map)
                        else:
                            saver_tmp = tf.train.Saver(restore_vars)

                        saver_tmp.restore(self.sess, path)
                    else:
                        raise err








    ############################################################
    # Public methods
    ############################################################

    def n_minibatch(self, n):
        return math.ceil(float(n) / self.minibatch_size)

    def minibatch_scale(self, n):
        return float(n) / self.minibatch_size

    def check_numerics(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for op in self.check_numerics_ops:
                    self.sess.run(op)

    def run_train_step(self, feed_dict, return_losses=True, return_reconstructions=False, return_labels=False):
        return NotImplementedError

    def evaluate_classifier(
            self,
            cv_data,
            X_cv=None,
            X_mask_cv=None,
            y_cv=None,
            y_mask_cv=None,
            segtype=None,
            ix2label=None,
            plot=True,
            verbose=True
    ):
        summary = ''
        eval_dict = {}
        binary = self.binary_classifier
        if segtype is None:
            segtype = self.segtype

        if self.task == 'utterance_classifier':
            if X_cv is None or X_mask_cv is None:
                X_cv, X_mask_cv = cv_data.inputs(
                    segments=segtype,
                    padding=self.input_padding,
                    normalize=self.normalize_data,
                    center=self.center_data,
                    resample=self.resample_inputs,
                    max_len=self.max_len
                )
            if y_cv is None or y_mask_cv is None:
                y_cv, y_mask_cv = cv_data.targets(
                    segments=segtype,
                    padding=self.target_padding,
                    reverse=self.reverse_targets,
                    normalize=self.normalize_data,
                    center=self.center_data,
                    resample=self.resample_outputs,
                    max_len=self.max_len
                )
            labels_cv = cv_data.labels(one_hot=False, segment_type=segtype)

            if self.speaker_emb_dim:
                speaker = cv_data.segments(segtype).speaker.as_matrix()
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    self.set_predict_mode(True)
                    if verbose:
                        sys.stderr.write('Predicting labels...\n\n')

                    if self.pad_seqs:
                        if not np.isfinite(self.eval_minibatch_size):
                            minibatch_size = len(y_cv)
                        else:
                            minibatch_size = self.eval_minibatch_size
                        n_minibatch = math.ceil(float(len(y_cv)) / minibatch_size)
                    else:
                        minibatch_size = 1
                        n_minibatch = len(y_cv)

                    to_run = []

                    labels_pred = []
                    to_run.append(self.labels_post)

                    if binary:
                        encoding = []
                        encoding_entropy = []
                        to_run += [self.encoding_post, self.encoding_entropy]
                    else:
                        encoding = None

                    for i in range(0, len(X_cv), minibatch_size):
                        if self.pad_seqs:
                            indices = np.arange(i, min(i + minibatch_size, len(X_cv)))
                        else:
                            indices = i

                        fd_minibatch = {
                            self.X: X_cv[indices],
                            self.X_mask: X_mask_cv[indices],
                            self.y: y_cv[indices],
                            self.y_mask: y_mask_cv[indices],
                            self.training: False
                        }

                        if self.speaker_emb_dim:
                            fd_minibatch[self.speaker] = speaker[indices]

                        out = self.sess.run(
                            to_run,
                            feed_dict=fd_minibatch
                        )

                        labels_pred_batch = out[0]
                        labels_pred.append(labels_pred_batch)

                        if binary:
                            encoding_batch, encoding_entropy_batch = out[1:]
                            encoding.append(encoding_batch)
                            encoding_entropy.append(encoding_entropy_batch)

                    labels_pred = np.concatenate(labels_pred, axis=0)

                    if binary:
                        encoding = np.concatenate(encoding, axis=0)
                        encoding_entropy = np.concatenate(encoding_entropy, axis=0).mean()

                    if plot:
                        if self.task == 'utterance_classifier' and ix2label is not None:
                            labels_string = np.vectorize(lambda x: ix2label[x])(labels_cv.astype('int'))

                            plot_label_heatmap(
                                labels_string,
                                labels_pred.astype('int'),
                                label_map=self.label_map,
                                dir=self.outdir
                            )

                            if binary:
                                if self.keep_plot_history:
                                    iter = self.global_step.eval(session=self.sess) + 1
                                    suffix = '_%d.png' % iter
                                else:
                                    suffix = '.png'

                                plot_binary_unit_heatmap(
                                    labels_string,
                                    encoding,
                                    label_map=self.label_map,
                                    dir=self.outdir,
                                    suffix=suffix
                                )

                    h, c, v = homogeneity_completeness_v_measure(labels_cv, labels_pred)
                    ami = adjusted_mutual_info_score(labels_cv, labels_pred)
                    fmi = fowlkes_mallows_score(labels_cv, labels_pred)

                    eval_dict['homogeneity'] = h
                    eval_dict['completeness'] = c
                    eval_dict['v_measure'] = v
                    eval_dict['ami'] = ami
                    eval_dict['fmi'] = fmi

                    pred_eval_summary = ''
                    if self.binary_classifier:
                        pred_eval_summary += 'Encoding entropy: %s\n\n' % encoding_entropy
                    pred_eval_summary +=  'Labeling scores (predictions):\n'
                    pred_eval_summary += '  Homogeneity:                 %s\n' % h
                    pred_eval_summary += '  Completeness:                %s\n' % c
                    pred_eval_summary += '  V-measure:                   %s\n' % v
                    pred_eval_summary += '  Adjusted mutual information: %s\n' % ami
                    pred_eval_summary += '  Fowlkes-Mallows index:       %s\n\n' % fmi

                    if verbose:
                        sys.stderr.write(pred_eval_summary)

                    if self.binary_classifier or not self.k:
                        if not self.k:
                            k = 2 ** self.emb_dim
                        else:
                            k = 2 ** self.k
                    else:
                        k = self.k

                    labels_rand = np.random.randint(0, k, labels_pred.shape)

                    h, c, v = homogeneity_completeness_v_measure(labels_cv, labels_rand)
                    ami = adjusted_mutual_info_score(labels_cv, labels_rand)
                    fmi = fowlkes_mallows_score(labels_cv, labels_rand)

                    rand_eval_summary = ''
                    if self.binary_classifier:
                        pred_eval_summary += 'Encoding entropy: %s\n\n' % encoding_entropy
                    rand_eval_summary += 'Labeling scores (predictions):\n'
                    rand_eval_summary += '  Homogeneity:                 %s\n' % h
                    rand_eval_summary += '  Completeness:                %s\n' % c
                    rand_eval_summary += '  V-measure:                   %s\n' % v
                    rand_eval_summary += '  Adjusted mutual information: %s\n' % ami
                    rand_eval_summary += '  Fowlkes-Mallows index:       %s\n\n' % fmi

                    if verbose:
                        sys.stderr.write(rand_eval_summary)

                    self.set_predict_mode(False)

                    summary += pred_eval_summary + rand_eval_summary
        else:
            if verbose:
                sys.stderr.write('The system is in segmentation mode and does not perform utterance classification. Skipping classifier evaluation...\n')
            labels_pred = None
            encoding = None

        return eval_dict, labels_pred, encoding, summary

    def classify_utterances(
            self,
            cv_data=None,
            X_cv=None,
            X_mask_cv=None,
            y_cv=None,
            y_mask_cv=None,
            segtype=None,
            ix2label=None,
            plot=True,
            verbose=True
    ):
        eval_dict = {}
        binary = self.binary_classifier
        if segtype is None:
            segtype = self.segtype

        if self.task == 'utterance_classifier':
            if X_cv is None or X_mask_cv is None:
                X_cv, X_mask_cv = cv_data.inputs(
                    segments=segtype,
                    padding=self.input_padding,
                    normalize=self.normalize_data,
                    center=self.center_data,
                    resample=self.resample_inputs,
                    max_len=self.max_len
                )
            if y_cv is None or y_mask_cv is None:
                y_cv, y_mask_cv = cv_data.targets(
                    segments=segtype,
                    padding=self.target_padding,
                    reverse=self.reverse_targets,
                    normalize=self.normalize_data,
                    center=self.center_data,
                    resample=self.resample_outputs,
                    max_len=self.max_len
                )

            segments = cv_data.segments(segment_type=segtype)
            segments.reset_index(inplace=True)

            classifier_scores, labels_pred, encoding, summary = self.evaluate_classifier(
                cv_data,
                X_cv=X_cv,
                X_mask_cv=X_mask_cv,
                y_cv=y_cv,
                y_mask_cv=y_mask_cv,
                segtype=segtype,
                ix2label=ix2label,
                plot=plot,
                verbose=verbose
            )
            eval_dict.update(classifier_scores)

            if binary:
                out_data = pd.DataFrame(encoding, columns=['d%d' % (i+1) for i in range(self.k)])
                out_data = pd.concat([segments, out_data], axis=1)
            else:
                out_data = None
        else:
            if verbose:
                sys.stderr.write('The system is in segmentation mode and does not perform utterance classification. Skipping classifier evaluation...\n')
            out_data = None

        return out_data, eval_dict, summary

    def run_evaluation(
            self,
            cv_data,
            X_cv=None,
            X_mask_cv=None,
            y_cv=None,
            y_mask_cv=None,
            n_plot=10,
            ix2label=None,
            y_means_cv=None,
            training_batch_norm=False,
            training_dropout=False,
            shuffle=False,
            segtype=None,
            plot=True,
            verbose=True
    ):
        eval_dict = {}

        binary = self.binary_classifier
        seg = 'hmlstm' in self.encoder_type.lower()
        if segtype is None:
            segtype = self.segtype

        if X_cv is None or X_mask_cv is None:
            X_cv, X_mask_cv = cv_data.inputs(
                segments=segtype,
                padding=self.input_padding,
                normalize=self.normalize_data,
                center=self.center_data,
                resample=self.resample_inputs,
                max_len=self.max_len
            )
        if y_cv is None or y_mask_cv is None:
            y_cv, y_mask_cv = cv_data.targets(
                segments=segtype,
                padding=self.target_padding,
                reverse=self.reverse_targets,
                normalize=self.normalize_data,
                center=self.center_data,
                resample=self.resample_outputs,
                max_len=self.max_len
            )
        labels_cv = cv_data.labels(one_hot=False, segment_type=segtype)

        if self.speaker_emb_dim:
            speaker = cv_data.segments(segtype).speaker.as_matrix()

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if self.pad_seqs:
                    if not np.isfinite(self.eval_minibatch_size):
                        minibatch_size = len(y_cv)
                    else:
                        minibatch_size = self.eval_minibatch_size
                    n_minibatch = math.ceil(float(len(y_cv)) / minibatch_size)
                else:
                    minibatch_size = 1
                    n_minibatch = len(y_cv)

                if self.task == 'utterance_classifier':
                    classifier_scores, labels_pred, encoding, _ = self.evaluate_classifier(
                        cv_data,
                        X_cv=X_cv,
                        X_mask_cv=X_mask_cv,
                        y_cv=y_cv,
                        y_mask_cv=y_mask_cv,
                        segtype=segtype,
                        ix2label=ix2label,
                        plot=plot,
                        verbose=verbose
                    )
                    eval_dict.update(classifier_scores)
                else:
                    if self.segment_eval_freq and self.global_step.eval(session=self.sess) % self.segment_eval_freq == 0:
                        if verbose:
                            sys.stderr.write('Extracting segmenter states...\n')

                        segmentation_probs = [[] for _ in self.segmentation_probs]
                        states = [[] for _ in self.segmentation_probs]

                        if shuffle:
                            perm, perm_inv = get_random_permutation(len(y_cv))

                        for i in range(0, len(X_cv), minibatch_size):
                            if shuffle:
                                if self.pad_seqs:
                                    indices = perm[i:i+minibatch_size]
                                else:
                                    indices = perm[i]
                            else:
                                if self.pad_seqs:
                                    indices = slice(i, i+minibatch_size, 1)
                                else:
                                    indices = i

                            fd_minibatch = {
                                self.X: X_cv[indices],
                                self.X_mask: X_mask_cv[indices],
                                self.y: y_cv[indices],
                                self.y_mask: y_mask_cv[indices],
                                self.training: training_batch_norm,
                                self.training: training_dropout
                            }

                            if self.speaker_emb_dim:
                                fd_minibatch[self.speaker] = speaker[indices]

                            segmentation_probs_cur, states_cur = self.sess.run(
                                [self.segmentation_probs, self.encoder_hidden_states[:-1]],
                                feed_dict=fd_minibatch
                            )

                            for j in range(len(self.segmentation_probs)):
                                segmentation_probs[j].append(segmentation_probs_cur[j])
                                states[j].append(states_cur[j])

                        new_segmentation_probs = []
                        new_states = []
                        for j in range(len(self.segmentation_probs)):
                            new_segmentation_probs.append(
                                np.concatenate(segmentation_probs[j], axis=0)
                            )
                            new_states.append(
                                np.concatenate(states[j], axis=0)
                            )
                        segmentation_probs = new_segmentation_probs
                        states = new_states

                        if verbose:
                            sys.stderr.write('Computing segmentation tables...\n')

                        if 'bsn' in self.encoder_boundary_activation.lower():
                            smoothing_algorithm = None
                            n_points = None
                        else:
                            smoothing_algorithm = 'rbf'
                            n_points = 1000

                        tables = cv_data.get_segment_tables_from_segmenter_states(
                            segmentation_probs,
                            parent_segment_type=segtype,
                            states=states,
                            state_activation=self.encoder_inner_activation,
                            algorithm=smoothing_algorithm,
                            algorithm_params=None,
                            offset=10,
                            n_points=n_points,
                            mask=X_mask_cv,
                            padding=self.input_padding
                        )

                        segmentation_scores = []

                        sys.stderr.write('\nSEGMENTATION EVAL:\n\n')
                        for i, f in enumerate(tables):
                            segmentation_scores.append({'phn': None, 'wrd': None})

                            sys.stderr.write('  Layer %s\n' %(i + 1))

                            s = cv_data.score_segmentation('phn', f, tol=0.02)[0]
                            B_P, B_R, B_F = f_measure(s['b_tp'], s['b_fp'], s['b_fn'])
                            W_P, W_R, W_F = f_measure(s['w_tp'], s['w_fp'], s['w_fn'])
                            sys.stderr.write('    Phonemes:\n')
                            sys.stderr.write('       B P: %s\n' %B_P)
                            sys.stderr.write('       B R: %s\n' %B_R)
                            sys.stderr.write('       B F: %s\n\n' %B_F)
                            sys.stderr.write('       W P: %s\n' %W_P)
                            sys.stderr.write('       W R: %s\n' %W_R)
                            sys.stderr.write('       W F: %s\n\n' %W_F)

                            segmentation_scores[-1]['phn'] = {
                                'b_p': B_P,
                                'b_r': B_R,
                                'b_f': B_F,
                                'w_p': W_P,
                                'w_r': W_R,
                                'w_f': W_F,
                            }

                            s = cv_data.score_segmentation('wrd', f, tol=0.03)[0]
                            B_P, B_R, B_F = f_measure(s['b_tp'], s['b_fp'], s['b_fn'])
                            W_P, W_R, W_F = f_measure(s['w_tp'], s['w_fp'], s['w_fn'])
                            sys.stderr.write('    Words:\n')
                            sys.stderr.write('       B P: %s\n' % B_P)
                            sys.stderr.write('       B R: %s\n' % B_R)
                            sys.stderr.write('       B F: %s\n\n' % B_F)
                            sys.stderr.write('       W P: %s\n' % W_P)
                            sys.stderr.write('       W R: %s\n' % W_R)
                            sys.stderr.write('       W F: %s\n\n' % W_F)

                            segmentation_scores[-1]['wrd'] = {
                                'b_p': B_P,
                                'b_r': B_R,
                                'b_f': B_F,
                                'w_p': W_P,
                                'w_r': W_R,
                                'w_f': W_F,
                            }

                            cv_data.dump_segmentations_to_textgrid(outdir=self.outdir, suffix='_l%d' % (i + 1), segments=f)

                if plot:
                    if verbose:
                        sys.stderr.write('Plotting...\n\n')

                    if self.task == 'utterance_classifier' and ix2label is not None:
                        labels_string = np.vectorize(lambda x: ix2label[x])(labels_cv.astype('int'))
                        titles = labels_string[self.plot_ix]
                    else:
                        titles = [None] * n_plot

                    to_run = [self.reconst]
                    if seg:
                        to_run += [self.segmentation_probs, self.encoder_hidden_states]

                    if self.pad_seqs:
                        X_cv_plot = X_cv[self.plot_ix]
                        y_cv_plot = (y_cv[self.plot_ix] * y_mask_cv[self.plot_ix][..., None]) if self.normalize_data else y_cv[self.plot_ix]

                        fd_minibatch = {
                            self.X: X_cv[self.plot_ix] if self.pad_seqs else [X_cv[ix] for ix in self.plot_ix],
                            self.X_mask: X_mask_cv[self.plot_ix] if self.pad_seqs else [X_mask_cv[ix] for ix in self.plot_ix],
                            self.y: y_cv[self.plot_ix] if self.pad_seqs else [y_cv[ix] for ix in self.plot_ix],
                            self.y_mask: y_mask_cv[self.plot_ix] if self.pad_seqs else [y_mask_cv[ix] for ix in self.plot_ix],
                            self.training: training_dropout,
                            self.training: training_dropout
                        }

                        if self.speaker_emb_dim:
                            fd_minibatch[self.speaker] = speaker[self.plot_ix]

                        out = self.sess.run(
                            to_run,
                            feed_dict=fd_minibatch
                        )
                    else:
                        X_cv_plot = [X_cv[ix][0] for ix in self.plot_ix]
                        y_cv_plot = [((y_cv[ix] * y_mask_cv[ix][..., None]) if self.normalize_data else y_cv[ix])[0] for ix in self.plot_ix]

                        out = []
                        for ix in self.plot_ix:
                            fd_minibatch = {
                                self.X: X_cv[ix],
                                self.X_mask: X_mask_cv[ix],
                                self.y: y_cv[ix],
                                self.y_mask: y_mask_cv[ix],
                                self.training: training_dropout,
                                self.training: training_dropout
                            }

                            out_cur = self.sess.run(
                                to_run,
                                feed_dict=fd_minibatch
                            )

                            for i, o in enumerate(out_cur):
                                if len(out) < len(out_cur):
                                    out.append([o[0]])
                                else:
                                    out[i].append(o[0])

                    reconst = out[0]

                    if seg:
                        if self.pad_seqs:
                            segmentation_probs = np.stack(out[1], axis=2)
                            states = out[2]
                        else:
                            segmentation_probs = []
                            for s in out[1]:
                                segmentation_probs.append(np.stack(s, axis=1))
                            states = out[2]
                    else:
                        segmentation_probs = None
                        states = None

                    if 'bsn' in self.encoder_boundary_activation:
                        hard_segmentations=True
                    else:
                        hard_segmentations=False

                    plot_acoustic_features(
                        X_cv_plot,
                        y_cv_plot,
                        reconst,
                        titles=titles,
                        segmentation_probs=segmentation_probs,
                        states=states,
                        hard_segmentations=hard_segmentations,
                        target_means=y_means_cv[self.plot_ix] if self.residual_decoder else None,
                        label_map=self.label_map,
                        dir=self.outdir
                    )

                    if self.task == 'utterance_classifier':
                        self.plot_label_histogram(labels_pred)

                self.set_predict_mode(False)

                return eval_dict

    def fit(
            self,
            train_data,
            cv_data=None,
            n_iter=None,
            ix2label=None,
            n_plot=10,
            verbose=True
    ):
        if self.global_step.eval(session=self.sess) == 0:
            self.save()

        n_fold = 512
        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        sys.stderr.write('Extracting training and cross-validation data...\n')
        t0 = time.time()

        if self.task == 'streaming_autoencoder':
            X, new_series = train_data.features(fold=n_fold, filter=None)
            n_train = len(X)
        else:
            X, X_mask = train_data.inputs(
                segments=self.segtype,
                padding=self.input_padding,
                max_len=self.max_len,
                normalize=self.normalize_data,
                center=self.center_data,
                resample=self.resample_inputs
            )
            y, y_mask = train_data.targets(
                segments=self.segtype,
                padding=self.target_padding,
                max_len=self.max_len,
                reverse=self.reverse_targets,
                normalize=self.normalize_data,
                center=self.center_data,
                resample=self.resample_outputs
            )
            n_train = len(y)

        if cv_data is None:
            if self.task == 'streaming_autoencoder':
                X_cv = X
            else:
                X_cv = X
                X_mask_cv = X_mask
                y_cv = y
                y_mask_cv = y_mask

            if self.plot_ix is None or len(self.plot_ix) != n_plot:
                self.plot_ix = np.random.choice(np.arange(len(X)), size=n_plot)
        else:
            if self.task == 'streaming_autoencoder':
                X_cv = cv_data.features()
            else:
                X_cv, X_mask_cv = cv_data.inputs(
                    segments=self.segtype,
                    padding=self.input_padding,
                    max_len=self.max_len,
                    normalize=self.normalize_data,
                    center=self.center_data,
                    resample=self.resample_inputs
                )
                y_cv, y_mask_cv = cv_data.targets(
                    segments=self.segtype,
                    padding=self.target_padding,
                    max_len=self.max_len,
                    reverse=self.reverse_targets,
                    normalize=self.normalize_data,
                    center=self.center_data,
                    resample=self.resample_outputs
                )

            if self.plot_ix is None or len(self.plot_ix) != n_plot:
                self.plot_ix = np.random.choice(np.arange(len(X_cv)), size=n_plot)

        t1 = time.time()

        sys.stderr.write('Training and cross-validation data extracted in %ds\n\n' % (t1 - t0))
        sys.stderr.flush()

        if self.residual_decoder:
            mean_axes = (0, 1)
            y_means = y.mean(axis=mean_axes, keepdims=True) * y_mask[..., None]
            y_means_cv = y_cv.mean(axis=mean_axes, keepdims=True) * y_mask_cv[..., None]
            y -= y_means
            y_cv -= y_means_cv
            if self.normalize_data:
                y -= y.min()
                y_cv -= y_cv.min()
                y = y / (y.max() + 2 * self.epsilon)
                y = y / (y.max() + 2 * self.epsilon)
                y_cv = y_cv / (y_cv.max() + 2 * self.epsilon)
                y *= y_mask[..., None]
                y_cv *= y_mask_cv[..., None]
                y += self.epsilon
                y_cv += self.epsilon
        else:
            y_means_cv = None

        if n_iter is None:
            n_iter = self.n_iter

        if self.speaker_emb_dim:
            speaker = train_data.segments(self.segtype).speaker.as_matrix()

        if verbose:
            sys.stderr.write('*' * 100 + '\n')
            sys.stderr.write(self.report_settings())
            sys.stderr.write('*' * 100 + '\n\n')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.pad_seqs:
                    if not np.isfinite(self.minibatch_size):
                        minibatch_size = n_train
                    else:
                        minibatch_size = self.minibatch_size
                    n_minibatch = math.ceil(float(n_train) / minibatch_size)
                else:
                    minibatch_size = 1
                    n_minibatch = n_train

                if self.task == 'streaming_autoencoder':
                    eval_dict = {}
                else:
                    eval_dict = self.run_evaluation(
                        cv_data if cv_data is not None else train_data,
                        X_cv=X_cv,
                        X_mask_cv=X_mask_cv,
                        y_cv=y_cv,
                        y_mask_cv=y_mask_cv,
                        ix2label=ix2label,
                        y_means_cv=None if y_means_cv is None else y_means_cv,
                        segtype=self.segtype,
                        plot=True,
                        verbose=verbose
                    )

                while self.global_step.eval(session=self.sess) < n_iter:
                    if self.task == 'streaming_autoencoder':
                        last_state = None
                        perm = np.arange(n_train)
                    else:
                        perm, perm_inv = get_random_permutation(n_train)

                    if verbose:
                        t0_iter = time.time()
                        sys.stderr.write('-' * 50 + '\n')
                        sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        sys.stderr.write('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                    if self.n_correspondence and self.global_step.eval(session=self.sess) + 1 >= self.correspondence_start_iter:
                        if verbose:
                            sys.stderr.write('Extracting and caching correspondence autoencoder targets...\n')
                        segment_embeddings, segment_spans = self._collect_previous_segments(self.n_correspondence, train_data, self.segtype, X=X, X_mask=X_mask)

                    if verbose:
                        sys.stderr.write('Updating...\n')
                        pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                    loss_total = 0.

                    for i in range(0, n_train, minibatch_size):
                        if self.pad_seqs:
                            indices = perm[i:i+minibatch_size]
                        else:
                            indices = perm[i]

                        fd_minibatch = {
                            self.training: True,
                            self.training: True
                        }

                        if self.task == 'streaming_autoencoder':
                            fd_minibatch[self.X] = X[int(indices)]
                            reset_state = new_series[int(indices)]

                            if self.speaker_emb_dim:
                                fd_minibatch[self.speaker] = speaker[int(indices)]

                            if reset_state or last_state is None:
                                state = self.sess.run(self.encoder_zero_state, feed_dict=fd_minibatch)
                            else:
                                state = last_state

                            fd_minibatch.update(self._map_state(state))

                        else:
                            fd_minibatch[self.X] = X[indices]
                            fd_minibatch[self.X_mask] = X_mask[indices]
                            fd_minibatch[self.y] = y[indices]
                            fd_minibatch[self.y_mask] = y_mask[indices]

                            if self.speaker_emb_dim:
                                fd_minibatch[self.speaker] = speaker[indices]

                            if self.n_correspondence and self.global_step.eval(session=self.sess) + 1 >= self.correspondence_start_iter:
                                for l in range(len(segment_embeddings)):
                                    fd_minibatch[self.correspondence_embedding_placeholders[l]] = segment_embeddings[l]
                                    fd_minibatch[self.correspondence_feature_placeholders[l]] = segment_spans[l]

                        info_dict = self.run_train_step(fd_minibatch)
                        loss_cur = info_dict['loss']
                        reg_cur = info_dict['regularizer_loss']

                        if self.ema_decay:
                            self.sess.run(self.ema_op)
                        if not np.isfinite(loss_cur):
                            loss_cur = 0
                        loss_total += loss_cur

                        self.sess.run(self.incr_global_batch_step)
                        if verbose:
                            pb.update((i/minibatch_size)+1, values=[('loss', loss_cur), ('reg', reg_cur)])

                        self.check_numerics()

                        if self.task == 'streaming_autoencoder':
                            X_plot = fd_minibatch[self.X]
                            x_len = X_plot.shape[1]

                            last_state = []
                            states = info_dict['encoder_states']
                            encoder_hidden_states = [states[i][1] for i in range(len(states))]
                            segmentations = [states[i][2] for i in range(len(states) - 1)]

                            for i in range(len(states)):
                                state_layer = []
                                for j in range(len(states[i])):
                                    state_layer.append(states[i][j][-1])
                                last_state.append(tuple(state_layer))
                            last_state = tuple(last_state)

                            if x_len > self.window_len_left + self.window_len_right:
                                ix = np.random.randint(self.window_len_left, x_len-self.window_len_right)
                                lb = max(-1, ix - self.window_len_left)
                                rb = min(x_len, ix + self.window_len_right + 1)
                                length_left = ix - lb
                                length_right = rb - (ix + 1)
                                preds_left = info_dict['out_left']
                                preds_right = info_dict['out_right']
                                segmentation_probs = []
                                for l in segmentations:
                                    l = np.swapaxes(l, 0, 1)
                                    segmentation_probs.append(l[:,:ix])
                                segmentation_probs = np.concatenate(segmentation_probs, axis=2)
                                states = []
                                for l in encoder_hidden_states:
                                    l = np.swapaxes(l, 0, 1)
                                    states.append(l[:,:ix])

                                y_plot_left = np.zeros((1, self.window_len_left, self.frame_dim))
                                y_plot_left[:, -length_left:] = X_plot[:,ix:lb:-1,:self.frame_dim]
                                y_plot_right = np.zeros((1, self.window_len_right, self.frame_dim))
                                y_plot_right[:, :length_right] = X_plot[:,ix+1:rb,:self.frame_dim]
                                reconst_left = preds_left[ix,:X_plot.shape[0]]
                                reconst_right = preds_right[ix,:X_plot.shape[0]]

                                plot_acoustic_features(
                                    X_plot[:, :ix],
                                    y_plot_left,
                                    reconst_left,
                                    segmentation_probs=segmentation_probs,
                                    states=states,
                                    drop_zeros=False,
                                    label_map=self.label_map,
                                    dir=self.outdir,
                                    suffix='_left.png'
                                )

                                plot_acoustic_features(
                                    X_plot[:, :ix],
                                    y_plot_right,
                                    reconst_right,
                                    segmentation_probs=segmentation_probs,
                                    states=states,
                                    drop_zeros=False,
                                    label_map=self.label_map,
                                    dir=self.outdir,
                                    suffix='_right.png'
                                )

                    loss_total /= n_minibatch

                    self.sess.run(self.incr_global_step)

                    if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        try:
                            self.check_numerics()
                            numerics_passed = True
                        except:
                            numerics_passed = False

                        if numerics_passed:
                            if verbose:
                                sys.stderr.write('Saving model...\n')

                            self.save()

                            if self.task == 'streaming_autoencoder':
                                eval_dict = {}
                            else:
                                eval_dict = self.run_evaluation(
                                    cv_data if cv_data is not None else train_data,
                                    X_cv=X_cv,
                                    X_mask_cv=X_mask_cv,
                                    y_cv=y_cv,
                                    y_mask_cv=y_mask_cv,
                                    ix2label=ix2label,
                                    y_means_cv=None if y_means_cv is None else y_means_cv,
                                    segtype=self.segtype,
                                    plot=True,
                                    verbose=verbose
                                )

                            fd_summary = {
                                self.loss_summary: loss_total
                            }

                            if self.task == 'utterance_classifier':
                                fd_summary[self.homogeneity] = eval_dict['homogeneity']
                                fd_summary[self.completeness] = eval_dict['completeness']
                                fd_summary[self.v_measure] = eval_dict['v_measure']
                                fd_summary[self.ami] = eval_dict['ami']
                                fd_summary[self.fmi] = eval_dict['fmi']


                            summary_metrics = self.sess.run(self.summary_metrics, feed_dict=fd_summary)
                            self.writer.add_summary(summary_metrics, self.global_step.eval(session=self.sess))

                            if 'segmentation_scores' in eval_dict and self.task != 'streaming_autoencoder' and 'hmlstm' in self.encoder_type.lower():
                                segmentation_scores = eval_dict['segmentation_scores']
                                for i in range(self.n_layers_encoder - 1):
                                    for s in ['phn', 'wrd']:
                                        fd_summary[self.segmentation_scores[i][s]['b_p']] = segmentation_scores[i][s]['b_p']
                                        fd_summary[self.segmentation_scores[i][s]['b_r']] = segmentation_scores[i][s]['b_r']
                                        fd_summary[self.segmentation_scores[i][s]['b_f']] = segmentation_scores[i][s]['b_f']
                                        fd_summary[self.segmentation_scores[i][s]['w_p']] = segmentation_scores[i][s]['w_p']
                                        fd_summary[self.segmentation_scores[i][s]['w_r']] = segmentation_scores[i][s]['w_r']
                                        fd_summary[self.segmentation_scores[i][s]['w_f']] = segmentation_scores[i][s]['w_f']

                                    summary_segmentations = self.sess.run(self.summary_segmentations, feed_dict=fd_summary)
                                    self.writer.add_summary(summary_segmentations, self.global_step.eval(session=self.sess))


                        else:
                            if verbose:
                                sys.stderr.write('Numerics check failed. Aborting save and reloading from previous checkpoint...\n')

                            self.load()

                    if verbose:
                        t1_iter = time.time()
                        time_str = pretty_print_seconds(t1_iter - t0_iter)
                        sys.stderr.write('Iteration time: %s\n' % time_str)

    def plot_label_histogram(self, labels_pred, dir=None):
        if dir is None:
            dir = self.outdir

        if self.binary_classifier or not self.k:
            if not self.k:
                bins = 2 ** self.emb_dim
            else:
                bins = 2 ** self.k
        else:
            bins = self.k

        if bins < 1000:
            plot_label_histogram(labels_pred, dir=dir, bins=bins)

    def plot_label_heatmap(self, labels, preds, dir=None):
        if dir is None:
            dir = self.outdir

        plot_label_heatmap(
            labels,
            preds,
            dir=dir,
            suffix=suffix
        )

    def initialized(self):
        """
        Check whether model has been initialized.

        :return: ``bool``; whether the model has been initialized.
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                uninitialized = self.sess.run(self.report_uninitialized)
                if len(uninitialized) == 0:
                    return True
                else:
                    return False

    def save(self, dir=None):
        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                failed = True
                i = 0

                # Try/except to handle race conditions in Windows
                while failed and i < 10:
                    try:
                        self.saver.save(self.sess, dir + '/model.ckpt')
                        with open(dir + '/m.obj', 'wb') as f:
                            pickle.dump(self, f)
                        failed = False
                    except:
                        sys.stderr.write('Write failure during save. Retrying...\n')
                        time.sleep(1)
                        i += 1
                if i >= 10:
                    sys.stderr.write('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.sess, dir + '/model_backup.ckpt')
                    with open(dir + '/m.obj', 'wb') as f:
                        pickle.dump(self, f)

    def load(self, outdir=None, predict=False, restore=True, allow_missing=True):
        """
        Load weights from a DNN-Seg checkpoint and/or initialize the DNN-Seg model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :param allow_missing: ``bool``; load all weights found in the checkpoint file, allowing those that are missing to remain at their initializations. If ``False``, weights in checkpoint must exactly match those in the model graph, or else an error will be raised. Leaving set to ``True`` is helpful for backward compatibility, setting to ``False`` can be helpful for debugging.
        :return:
        """
        if outdir is None:
            outdir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not self.initialized():
                    self.sess.run(tf.global_variables_initializer())
                    tf.tables_initializer().run()
                if restore and os.path.exists(outdir + '/checkpoint'):
                    self._restore_inner(outdir + '/model.ckpt', predict=predict, allow_missing=allow_missing)
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def set_predict_mode(self, mode):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not mode or self.ema_decay:
                    self.load(predict=mode)

    def report_settings(self, indent=0):
        out = ' ' * indent + 'MODEL SETTINGS:\n'
        out += ' ' * (indent + 2) + 'k: %s\n' %self.k
        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out

    def summary(self):
        out = ''


class AcousticEncoderDecoderMLE(AcousticEncoderDecoder):
    _INITIALIZATION_KWARGS = UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS

    _doc_header = """
        MLE implementation of unsupervised word classifier.

    """
    _doc_args = AcousticEncoderDecoder._doc_args
    _doc_kwargs = AcousticEncoderDecoder._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value,
                                                                                                  str) else "'%s'" % x.default_value)
                                     for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    def __init__(self, k, train_data, **kwargs):
        super(AcousticEncoderDecoderMLE, self).__init__(
            k,
            train_data,
            **kwargs
        )

        for kwarg in AcousticEncoderDecoderMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        kwarg_keys = [x.key for x in AcousticEncoderDecoder._INITIALIZATION_KWARGS]
        for kwarg_key in kwargs:
            if kwarg_key not in kwarg_keys:
                raise TypeError('__init__() got an unexpected keyword argument %s' %kwarg_key)

        self._initialize_metadata()

    def _initialize_metadata(self):
        super(AcousticEncoderDecoderMLE, self)._initialize_metadata()

    def _pack_metadata(self):
        md = super(AcousticEncoderDecoderMLE, self)._pack_metadata()

        for kwarg in AcousticEncoderDecoderMLE._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        super(AcousticEncoderDecoderMLE, self)._unpack_metadata(md)

        for kwarg in AcousticEncoderDecoderMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            sys.stderr.write('Saved model contained unrecognized attributes %s which are being ignored\n' %sorted(list(md.keys())))

    def _initialize_classifier(self, classifier_in):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                encoding_logits = classifier_in[..., :self.k]

                if self.task == 'utterance_classifier':
                    if self.binary_classifier:
                        if self.state_slope_annealing_rate:
                            rate = self.state_slope_annealing_rate
                            if self.slope_annealing_max is None:
                                slope_coef = 1 + rate * tf.cast(self.global_step, dtype=tf.float32)
                            else:
                                slope_coef = tf.minimum(self.slope_annealing_max, 1 + rate * tf.cast(self.global_step, dtype=tf.float32))
                            encoding_logits *= slope_coef
                        encoding_probs = tf.sigmoid(encoding_logits)
                        encoding = encoding_probs
                        encoding = get_activation(
                            self.encoder_state_discretizer,
                            training=self.training,
                            session=self.sess,
                            from_logits=False
                        )(encoding)
                        self.encoding_entropy = binary_entropy(encoding_logits, from_logits=True, session=self.sess)
                        self.encoding_entropy_mean = tf.reduce_mean(self.encoding_entropy)
                        self._regularize(encoding_logits, self.entropy_regularizer)
                        self._regularize(encoding, self.boundary_regularizer)
                    else:
                        encoding = tf.nn.softmax(encoding_logits)
                else:
                    encoding = encoding_logits

                return encoding

    def _initialize_output_model(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.task != 'streaming_autoencoder':
                    if self.normalize_data and self.constrain_output:
                        self.out = tf.sigmoid(self.decoder)
                    else:
                        self.out = self.decoder

                    # if self.n_correspondence:
                    #     self.correspondence_autoencoders = []
                    #     for l in range(self.n_layers_encoder - 1):
                    #         correspondence_autoencoder = self._initialize_decoder(self.encoder_hidden_states[l], self.resample_correspondence)
                    #         self.correspondence_autoencoders.append(correspondence_autoencoder)

    def _get_loss(self, targets, preds, use_dtw=False, layerwise=False, log_loss=False, mask=None, weights=None, reduce=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                loss = 0
                if use_dtw:
                    if layerwise:
                        for i in range(self.n_layers_decoder):
                            pred = self.decoder_layers[i]
                            if i < self.n_layers_decoder - 1:
                                targ = self.encoder_hidden_states[self.n_layers_decoder - i - 2]
                            else:
                                targ = self.y
                            loss_cur = self._soft_dtw_B(targ, pred, self.dtw_gamma, mask=mask)
                            if weights is not None:
                                loss_cur *= weights
                            if reduce:
                                loss_cur = tf.reduce_mean(loss_cur)
                            loss += loss_cur
                    else:
                        loss_cur = self._soft_dtw_B(targets, preds, self.dtw_gamma, mask=mask)
                        if weights is not None:
                            loss_cur *= weights
                        if reduce:
                            loss_cur = tf.reduce_mean(loss_cur)
                        loss += loss_cur
                else:
                    if log_loss:
                        if mask is not None:
                            loss += tf.losses.log_loss(targets, preds, weights=mask[..., None])
                        else:
                            loss += tf.losses.log_loss(targets, preds)
                    else:
                        if layerwise:
                            for i in range(self.n_layers_decoder):
                                pred = self.decoder_layers[i]
                                if i < self.n_layers_decoder - 1:
                                    targ = self.encoder_hidden_states[self.n_layers_decoder - i - 2]
                                else:
                                    targ = targets
                                loss_cur = (targ - pred) ** 2
                                if mask is not None:
                                    while len(mask.shape) < len(loss_cur.shape):
                                        mask = mask[..., None]
                                    loss_cur *= mask
                                if weights is not None:
                                    while len(weights.shape) < len(loss_cur.shape):
                                        weights = weights[..., None]
                                    loss_cur *= weights
                                if reduce:
                                    loss += tf.reduce_mean(loss_cur)
                                else:
                                    loss += loss_cur
                        else:
                            loss_cur = (targets - preds) ** 2
                            if mask is not None:
                                while len(mask.shape) < len(loss_cur.shape):
                                    mask = mask[..., None]
                                loss_cur *= mask
                            if weights is not None:
                                while len(weights.shape) < len(loss_cur.shape):
                                    weights = weights[..., None]
                                loss_cur *= weights
                            if reduce:
                                loss_cur = tf.reduce_mean(loss_cur)
                            loss += loss_cur

                return loss


    def _initialize_objective(self, n_train):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Define access points to important layers

                IMPLEMENTATION = 2
                K = 5

                if self.task == 'streaming_autoencoder':
                    units_utt = self.k
                    if self.emb_dim:
                        units_utt += self.emb_dim

                    self.encoder_cell = HMLSTMCell(
                        self.units_encoder + [units_utt],
                        self.n_layers_encoder,
                        training=self.training,
                        activation=self.encoder_inner_activation,
                        inner_activation=self.encoder_inner_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        boundary_activation=self.encoder_boundary_activation,
                        bottomup_regularizer=self.encoder_weight_regularization,
                        recurrent_regularizer=self.encoder_weight_regularization,
                        topdown_regularizer=self.encoder_weight_regularization,
                        boundary_regularizer=self.encoder_weight_regularization,
                        bias_regularizer=None,
                        layer_normalization=self.encoder_layer_normalization,
                        refeed_boundary=True,
                        power=self.boundary_power,
                        boundary_slope_annealing_rate=self.boundary_slope_annealing_rate,
                        state_slope_annealing_rate=self.state_slope_annealing_rate,
                        slope_annealing_max=self.slope_annealing_max,
                        state_discretizer=self.encoder_state_discretizer,
                        global_step=self.global_step,
                        implementation=2
                    )

                    self.encoder_zero_state = self.encoder_cell.zero_state(tf.shape(self.X)[0], self.FLOAT_TF)
                    encoder_state = []
                    for i in range(len(self.encoder_zero_state)):
                        state_layer = []
                        for j in range(len(self.encoder_zero_state[i])):
                            state_layer.append(tf.placeholder(self.FLOAT_TF, shape=self.encoder_zero_state[i][j].shape))
                        encoder_state.append(tuple(state_layer))

                    self.encoder_state = tuple(encoder_state)
                    loss, self.encoder_states, self.out_left, self.out_right = self._streaming_dynamic_scan(self.X)

                    self.encoder_hidden_states = [self.encoder_states[i][1] for i in range(len(self.encoder_states))]
                    self.segmentation_probs = [self.encoder_states[i][2] for i in range(len(self.encoder_states) - 1)]

                else:
                    self.reconst = self.out
                    if not self.dtw_gamma:
                        self.reconst *= self.y_mask[..., None]

                    self.encoding_post = self.encoding
                    if self.task == 'utterance_classifier':
                        self.labels_post = self.labels
                        self.label_probs_post = self.label_probs

                    targets = self.y
                    preds = self.out

                    loss = self._get_loss(
                        targets,
                        preds,
                        use_dtw=self.use_dtw,
                        layerwise='layerwise' in self.decoder_type.lower(),
                        log_loss=self.normalize_data and self.constrain_output,
                        mask=self.y_mask if self.mask_padding else None
                    )

                    if self.n_correspondence:
                        for l in range(self.n_layers_encoder - 1):
                            def compute_loss_cur():
                                correspondence_autoencoder = self._initialize_decoder(self.encoder_hidden_states[l], self.resample_correspondence)

                                targets = tf.expand_dims(tf.expand_dims(self.correspondence_feature_placeholders[l], axis=0), axis=0)
                                preds = correspondence_autoencoder

                                embeddings_by_timestep = self.encoder_hidden_states[l]
                                embeddings_by_timestep /= (tf.norm(embeddings_by_timestep, axis=-1, keepdims=True) + self.epsilon)

                                embeddings_by_segment = self.correspondence_embedding_placeholders[l]
                                embeddings_by_segment /= (tf.norm(embeddings_by_segment, axis=-1, keepdims=True) + self.epsilon)

                                cos_sim = tf.tensordot(embeddings_by_timestep, tf.transpose(embeddings_by_segment, perm=[1,0]), axes=1)
                                
                                alpha = 10
                                weights = cos_sim

                                if IMPLEMENTATION == 1:
                                    weights = tf.nn.softmax(weights * alpha, axis=-1)
                                    weights = tf.expand_dims(tf.expand_dims(weights, -1), -1)
                                    targets = tf.reduce_sum(targets * weights, axis=2)

                                    loss_cur = self._get_loss(
                                        targets,
                                        preds,
                                        use_dtw=self.use_dtw,
                                        log_loss=self.normalize_data and self.constrain_output,
                                        reduce=False
                                    )
                                elif IMPLEMENTATION == 2:
                                    ix = tf.argmax(weights, axis=-1)
                                    targets = tf.gather(self.correspondence_feature_placeholders[l], ix)

                                    loss_cur = self._get_loss(
                                        targets,
                                        preds,
                                        use_dtw=self.use_dtw,
                                        log_loss=self.normalize_data and self.constrain_output,
                                        reduce=False
                                    )
                                    # loss_cur = 0
                                    # _, k_best = tf.nn.top_k(weights, k=K)
                                    # for k in range(K):
                                    #     ix = k_best[..., k]
                                    #     targets = tf.gather(self.X_correspondence[l], ix)
                                    #
                                    #     loss_cur += self._get_loss(
                                    #         targets,
                                    #         preds,
                                    #         use_dtw=self.use_dtw,
                                    #         log_loss=self.normalize_data and self.constrain_output,
                                    #         reduce=False
                                    #     )
                                else:
                                    weights = tf.nn.softmax(weights * alpha, axis=-1)
                                    preds = tf.expand_dims(preds, axis=2)

                                    loss_cur = self._get_loss(
                                        targets,
                                        preds,
                                        use_dtw=self.use_dtw,
                                        log_loss=self.normalize_data and self.constrain_output,
                                        weights=weights,
                                        reduce=False
                                    )
                                    loss_cur = tf.reduce_sum(loss_cur, axis=2)

                                seg_probs = self.segmentation_probs[l]
                                while len(seg_probs.shape) < len(loss_cur.shape):
                                    seg_probs = seg_probs[..., None]
                                loss_cur *= seg_probs
                                loss_cur = tf.reduce_mean(loss_cur) * self.correspondence_loss_weight

                                return loss_cur

                            loss_cur = tf.cond(self.global_step + 1 >= self.correspondence_start_iter, compute_loss_cur, lambda: 0.)

                            self.regularizer_losses.append(loss_cur)

                if len(self.regularizer_losses) > 0:
                    self.regularizer_loss_total = tf.add_n(self.regularizer_losses)
                    loss += self.regularizer_loss_total
                else:
                    self.regularizer_loss_total = tf.constant(0., dtype=self.FLOAT_TF)

                self.loss = loss
                self.optim = self._initialize_optimizer(self.optim_name)
                self.train_op = self.optim.minimize(self.loss, global_step=self.global_batch_step)

    def run_train_step(
            self,
            feed_dict,
            return_loss=True,
            return_regularizer_loss=True,
            return_reconstructions=False,
            return_labels=False,
            return_label_probs=False,
            return_encoding_entropy=False,
            return_segmentation_probs=False
    ):
        out_dict = {}

        if return_loss or return_reconstructions or return_labels or return_label_probs:
            to_run = [self.train_op]
            to_run_names = []
            if return_loss:
                to_run.append(self.loss)
                to_run_names.append('loss')
            if return_regularizer_loss:
                to_run.append(self.regularizer_loss_total)
                to_run_names.append('regularizer_loss')
            if return_reconstructions:
                to_run.append(self.reconst)
                to_run_names.append('reconst')
            if return_labels:
                to_run.append(self.labels)
                to_run_names.append('labels')
            if return_label_probs:
                to_run.append(self.label_probs)
                to_run_names.append('label_probs')
            if return_encoding_entropy:
                to_run.append(self.encoding_entropy_mean)
                to_run_names.append('encoding_entropy')
            if self.encoder_type.lower() in ['cnn_hmlstm', 'hmlstm'] and return_segmentation_probs:
                to_run.append(self.segmentation_probs)
                to_run_names.append('segmentation_probs')
            if self.task == 'streaming_autoencoder':
                to_run.append(self.encoder_states)
                to_run.append(self.segmentation_probs)
                to_run.append(self.out_left)
                to_run.append(self.out_right)
                to_run_names.append('encoder_states')
                to_run_names.append('segmentation_probs')
                to_run_names.append('out_left')
                to_run_names.append('out_right')

            output = self.sess.run(to_run, feed_dict=feed_dict)

            for i, x in enumerate(output[1:]):
                out_dict[to_run_names[i]] = x

        return out_dict

    def report_settings(self, indent=0):
        out = super(AcousticEncoderDecoderMLE, self).report_settings(indent=indent)
        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out





