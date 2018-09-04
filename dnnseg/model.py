import sys
import os
import math
import time
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

from .data import get_random_permutation
from .kwargs import UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS
from .plot import plot_acoustic_features, plot_label_histogram, plot_label_heatmap, plot_binary_unit_heatmap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


def dense_layer(
        inputs,
        units,
        training,
        activation=None,
        batch_normalize=True,
        use_bias=True,
        session=None
):
    if session is None:
        session = tf.get_default_session()
    with session.as_default():
        with session.graph.as_default():
            x = inputs

            H = tf.layers.dense(x, units, use_bias=use_bias)
            if batch_normalize:
                H = tf.layers.batch_normalization(H, training=training)
            if activation is not None:
                H = activation(H)

            return H


def dense_residual_layer(
        inputs,
        training,
        units=None,
        layers_inner=2,
        activation_inner=tf.nn.elu,
        activation_outer=None,
        batch_normalize=True,
        use_bias=True,
        session=None
):
    if session is None:
        session = tf.get_default_session()
    with session.as_default():
        with session.graph.as_default():
            if units is None:
                units = inputs.shape[-1]
            if units == inputs.shape[-1]:
                x = inputs
            else:
                x = tf.layers.dense(inputs, units)

            F = x
            for i in range(layers_inner - 1):
                F = tf.layers.dense(F, units)
                if batch_normalize:
                    F = tf.layers.batch_normalization(F, training=training)
                if activation_inner is not None:
                    F = activation_inner(F)

            F = tf.layers.dense(F, units, use_bias=use_bias)
            if batch_normalize:
                F = tf.layers.batch_normalization(F, training=training)
            # if activation_inner is not None:
            #     F = activation_inner(F)

            H = F + x
            if activation_outer is not None:
                H = activation_outer(H)

            return H


class UnsupervisedWordClassifier(object):

    _INITIALIZATION_KWARGS = UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for unsupervised word classifier. Bayesian and MLE implementations inherit from ``UnsupervisedWordClassifier``.
        ``UnsupervisedWordClassifier`` is not a complete implementation and cannot be instantiated.

    """
    _doc_args = """
        :param k: ``int``; dimensionality of classifier.
    \n"""
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (
                                 x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value)
                             for
                             x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    def __new__(cls, *args, **kwargs):
        if cls is UnsupervisedWordClassifier:
            raise TypeError("UnsupervisedWordClassifier is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, k, **kwargs):

        self.k = k
        for kwarg in UnsupervisedWordClassifier._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        assert not (self.unroll and self.n_timesteps is None), 'Number of timesteps must be prespecified for an unrolled RNN'
        assert not (self.encoder_type == 'dense' and self.n_timesteps is None), 'Number of timesteps must be prespecified for a dense encoder'
        assert not (self.encoder_type == 'cnn' and self.n_timesteps is None), 'Number of timesteps must be prespecified for a CNN encoder'
        assert not (self.decoder_type == 'dense' and self.n_timesteps is None), 'Number of timesteps must be prespecified for a dense decoder'
        assert not (self.decoder_type == 'cnn' and self.n_timesteps is None), 'Number of timesteps must be prespecified for a CNN decoder'
        assert self.encoder_type in ['rnn', 'cnn', 'dense', 'dense_resnet'], 'Currently supported encoder types are "rnn", "cnn", "dense", and "dense_resnet"'
        assert self.decoder_type in ['rnn', 'cnn', 'dense', 'dense_resnet'], 'Currently supported decoder types are "rnn", "cnn", "dense", and "dense_resnet"'

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

    def _pack_metadata(self):
        md = {
            'k': self.k,
        }
        for kwarg in UnsupervisedWordClassifier._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.k = md.pop('k')
        for kwarg in UnsupervisedWordClassifier._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    def __getstate__(self):
        return self._pack_metadata()

    def __setstate__(self, state):
        self._unpack_metadata(state)
        self._initialize_session()
        self._initialize_metadata()

    def build(self, n_train, outdir=None, restore=True, verbose=True):
        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './dnnseg_model/'
        else:
            self.outdir = outdir

        self._initialize_inputs()
        self._initialize_encoder()
        self._initialize_classifier()
        self._augment_encoding()
        self._initialize_decoder()
        self._initialize_output_model()
        self._initialize_objective(n_train)
        self._initialize_ema()
        self._initialize_saver()
        self._initialize_logging()
        self.load(restore=restore)

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.X = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps, self.n_coef * (self.order + 1)), name='X')
                self.X_mask = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps), name='X_mask')

                self.X_feat_mean = tf.reduce_sum(self.X, axis=-2) / tf.reduce_sum(self.X_mask, axis=-1, keepdims=True)
                self.X_time_mean = tf.reduce_mean(self.X, axis=-1)

                if self.reconstruct_deltas:
                    self.frame_dim = self.n_coef * (self.order + 1)
                else:
                    self.frame_dim = self.n_coef
                self.y = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps, self.frame_dim), name='y')

                self.y_mask = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps), name='y_mask')

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
                self.batch_len = tf.shape(self.X)[0]

                self.loss_summary = tf.placeholder(tf.float32, name='loss_summary')
                self.homogeneity = tf.placeholder(tf.float32, name='homogeneity')
                self.completeness = tf.placeholder(tf.float32, name='completeness')
                self.v_measure = tf.placeholder(tf.float32, name='v_measure')

                self.training_src = tf.Variable(1, dtype=self.INT_TF, trainable=False, name='training')
                self.training = tf.cast(self.training_src, dtype=tf.bool)
                self.training_on = tf.assign(self.training_src, 1)
                self.training_off = tf.assign(self.training_src, 0)

    def _initialize_encoder(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.k
                if self.emb_dim:
                    units += self.emb_dim

                if self.encoder_type == 'rnn':
                    encoder = self.X
                    for i in range(self.n_layers_encoder - 1):
                        encoder = tf.keras.layers.GRU(
                            units,
                            return_sequences=True,
                            unroll=self.unroll,
                            recurrent_activation=tf.tanh,
                            activation=tf.sigmoid
                        )(encoder)
                    encoder = tf.keras.layers.GRU(
                        units,
                        return_sequences=False,
                        unroll=self.unroll,
                        recurrent_activation=tf.tanh
                    )(encoder)

                elif self.encoder_type == 'cnn':
                    encoder = tf.keras.layers.Conv2D(self.conv_n_filters, self.conv_kernel_size, strides=2, padding='same', activation='elu')(self.X[...,None])
                    encoder = tf.keras.layers.Conv2D(self.conv_n_filters * 2, self.conv_kernel_size, strides=2, padding='same', activation='elu')(encoder)
                    encoder = tf.layers.dense(tf.layers.Flatten()(encoder), self.k)

                elif self.encoder_type == 'dense':
                    encoder = tf.layers.Flatten()(self.X)
                    for _ in range(self.n_layers_encoder - 1):
                        encoder = tf.layers.dense(encoder, self.n_timesteps * self.frame_dim, activation=tf.nn.elu)
                    encoder = tf.layers.dense(encoder, units)

                elif self.encoder_type == 'dense_resnet':
                    encoder = tf.layers.Flatten()(self.X)
                    for i in range(self.n_layers_encoder - 1):
                        if i == 0:
                            encoder = dense_layer(
                                encoder,
                                self.n_timesteps * self.frame_dim,
                                self.training,
                                activation=tf.nn.elu,
                                batch_normalize=self.batch_normalize,
                                session=self.sess
                            )
                        else:
                            encoder = dense_residual_layer(
                                encoder,
                                self.training,
                                units=self.n_timesteps * self.frame_dim,
                                layers_inner=self.resnet_n_layers_inner,
                                activation_inner=tf.nn.elu,
                                activation_outer=None,
                                batch_normalize=self.batch_normalize,
                                session=self.sess
                            )
                    encoder = tf.layers.dense(encoder, units)

                self.encoder = encoder

    def _initialize_classifier(self):
        self.encoding = None
        raise NotImplementedError

    def _augment_encoding(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                if self.binary_classifier:
                    self.labels = self._binary2integer(tf.round(self.encoding))
                    self.label_probs = self._bernoulli2categorical(self.encoding)
                else:
                    self.labels = tf.argmax(self.encoding, axis=-1)
                    self.label_probs = self.encoding

                extra_dims = None

                if self.emb_dim:
                    extra_dims = tf.nn.elu(self.encoder[:,self.k:])

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

                self.extra_dims = extra_dims

                if self.extra_dims is not None:
                    self.decoder_in = tf.concat([self.encoding, self.extra_dims], axis=1)
                else:
                    self.decoder_in = self.encoding

    def _initialize_decoder(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.decoder_type == 'rnn':
                    self.decoder_in = tf.tile(
                        self.decoder_in[..., None, :],
                        [1, tf.shape(self.y)[1], 1]
                    ) * self.y_mask[...,None]

                    decoder = tf.keras.layers.LSTM(
                        self.y.shape[-1],
                        recurrent_activation='sigmoid',
                        return_sequences=True,
                        unroll=self.unroll
                    )(self.decoder_in)

                elif self.decoder_type == 'cnn':
                    assert self.n_timesteps is not None, 'n_timesteps must be defined when decoder_type == "cnn"'

                    decoder = tf.layers.dense(self.decoder_in, self.n_timesteps * self.frame_dim)[..., None]
                    decoder = tf.reshape(decoder, (self.batch_len, self.n_timesteps, self.frame_dim, 1))
                    decoder = tf.keras.layers.Conv2D(self.conv_n_filters, self.conv_kernel_size, padding='same', activation='elu')(decoder)
                    decoder = tf.keras.layers.Conv2D(1, self.conv_kernel_size, padding='same', activation='linear')(decoder)
                    decoder = tf.squeeze(decoder, axis=-1)

                elif self.decoder_type in ['dense', 'dense_resnet']:
                    n_classes = int(2 ** self.k) if self.binary_classifier else int(self.k)

                    # First layer
                    if self.per_class_decoder:
                        if self.extra_dims is not None:
                            decoder = tf.tile(
                                self.extra_dims[:, None, :],
                                [1, n_classes, 1]
                            )

                            decoder = dense_layer(
                                decoder,
                                self.n_timesteps * self.frame_dim,
                                self.training,
                                activation=tf.nn.elu,
                                batch_normalize=self.batch_normalize,
                                session=self.sess
                            )
                        else:
                            decoder = tf.Variable(
                                tf.random_uniform(shape=[1, n_classes, self.n_timesteps * self.frame_dim]),
                                dtype=self.FLOAT_TF
                            )
                    else:
                        decoder = dense_layer(
                            self.decoder_in,
                            self.n_timesteps * self.frame_dim,
                            self.training,
                            activation=tf.nn.elu,
                            batch_normalize=self.batch_normalize,
                            session=self.sess
                        )

                    # Intermediate layers
                    if self.decoder_type == 'dense':
                        for i in range(1, self.n_layers_decoder - 1):
                            decoder = dense_layer(
                                decoder,
                                self.n_timesteps * self.frame_dim,
                                self.training,
                                activation=tf.nn.elu,
                                batch_normalize=self.batch_normalize,
                                session=self.sess
                            )
                    else: # self.decoder_type = 'dense_resnet'
                        for i in range(1, self.n_layers_decoder - 1):
                            decoder = dense_residual_layer(
                                decoder,
                                self.training,
                                units=self.n_timesteps * self.frame_dim,
                                layers_inner=self.resnet_n_layers_inner,
                                activation_inner=tf.nn.elu,
                                activation_outer=None,
                                batch_normalize=self.batch_normalize,
                                session=self.sess
                            )

                    # Last layer
                    if self.n_layers_decoder > 1:
                        decoder = dense_layer(
                            decoder,
                            self.n_timesteps * self.frame_dim,
                            self.training,
                            activation=None,
                            batch_normalize=False,
                            session=self.sess
                        )

                    # Weight sum of class outputs, if applicable
                    if self.per_class_decoder:
                        decoder = decoder * self.label_probs[..., None]
                        decoder = tf.reduce_sum(
                            decoder,
                            axis=1
                        )

                    # Reshape
                    decoder = tf.reshape(decoder, (self.batch_len, self.n_timesteps, self.frame_dim))

                else:
                    raise ValueError('Decoder type "%s" not supported at this time' %self.decoder_type)

                self.decoder = decoder

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

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                tf.summary.scalar('loss_summary', self.loss_summary, collections=['metrics'])
                tf.summary.scalar('homogeneity', self.homogeneity, collections=['metrics'])
                tf.summary.scalar('completeness', self.completeness, collections=['metrics'])
                tf.summary.scalar('v_measure', self.v_measure, collections=['metrics'])
                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dnnseg', self.sess.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dnnseg')
                self.summary_metrics = tf.summary.merge_all(key='metrics')

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

                self.check_numerics_ops = [tf.check_numerics(v, 'Numerics check failed') for v in tf.trainable_variables()]

    def _initialize_ema(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                vars = tf.get_collection('trainable_variables')
                self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
                self.ema_op = self.ema.apply(vars)
                self.ema_map = {}
                for v in vars:
                    self.ema_map[self.ema.average_name(v)] = v
                self.ema_saver = tf.train.Saver(self.ema_map)

    def _binary2integer(self, b):
        k = int(b.shape[-1])
        if self.int_type.endswith('128'):
            assert k <= int(self.int_type[-3:]) / 2, 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' %(b.shape[-1], self.int_type)
        else:
            assert k <= int(self.int_type[-2:]) / 2, 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' %(b.shape[-1], self.int_type)
        base2 = 2 ** tf.range(k-1, limit=-1, delta=-1, dtype=self.INT_TF)
        while len(base2.shape) < len(b.shape):
            base2 = tf.expand_dims(base2, 0)

        return tf.reduce_sum(tf.cast(b, dtype=self.INT_TF) * base2, axis=-1)

    def _bernoulli2categorical(self, b):
        k = int(b.shape[-1])
        if self.int_type.endswith('128'):
            assert k <= int(self.int_type[-3:]), 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' % (b.shape[-1], self.int_type)
        else:
            assert k <= int(self.int_type[-2:]), 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' % (b.shape[-1], self.int_type)

        binary_matrix = tf.constant((np.expand_dims(np.arange(2 ** k), -1) & (1 << np.arange(k))).astype(bool).astype(int).T, dtype=self.FLOAT_TF)
        c = tf.expand_dims(b, -1) * binary_matrix + (1 - tf.expand_dims(b, -1)) * (1 - binary_matrix)
        c = tf.reduce_prod(c, -2)
        return c

    def _flat_matrix_diagonal_indices(self, n):
        out = np.arange(n) + np.arange(n) * n
        out = out.astype('int')

        return out

    def _flat_matrix_off_diagonal_indices(self, n):
        offset = np.zeros(n**2 - n)
        offset[np.arange(n-1) * 10] = 1
        offset = offset.cumsum()

        out = (np.arange(n**2 - n) + offset).astype('int')

        return out

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

    def run_incremental_evaluation(
            self,
            X_cv,
            X_mask_cv,
            y_cv,
            y_mask_cv,
            labels_cv,
            n_plot=10,
            ix2label=None,
            y_means_cv=None,
            verbose=True
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if verbose:
                    sys.stderr.write('Extracting predictions...\n')
                reconst = []
                labels_pred = []
                to_run = [self.reconst, self.labels_post]
                if self.binary_classifier:
                    encoding = []
                    to_run.append(self.encoding_post)
                for i in range(0, len(X_cv), self.eval_minibatch_size):
                    out = self.sess.run(
                        to_run,
                        feed_dict={
                            self.X: X_cv[i:i + self.eval_minibatch_size],
                            self.X_mask: X_mask_cv[i:i + self.eval_minibatch_size],
                            self.y: y_cv[i:i + self.eval_minibatch_size],
                            self.y_mask: y_mask_cv[i:i + self.eval_minibatch_size],
                        }
                    )
                    if self.binary_classifier:
                        reconst_batch, labels_pred_batch, encoding_batch = out
                        encoding.append(encoding_batch)
                    else:
                        reconst_batch, labels_pred_batch = out
                    reconst.append(reconst_batch)
                    labels_pred.append(labels_pred_batch)

                reconst = np.concatenate(reconst, axis=0)
                labels_pred = np.concatenate(labels_pred, axis=0)
                if self.binary_classifier:
                    encoding = np.concatenate(encoding, axis=0)

                if verbose:
                    sys.stderr.write('Plotting...\n')

                if ix2label is not None:
                    labels_string = np.vectorize(lambda x: ix2label[x])(labels_cv.astype('int'))
                    titles = labels_string[self.plot_ix]

                    plot_label_heatmap(
                        labels_string,
                        labels_pred.astype('int'),
                        dir=self.outdir
                    )
                    if self.binary_classifier:
                        plot_binary_unit_heatmap(
                            labels_string,
                            encoding,
                            dir=self.outdir
                        )

                else:
                    titles = [None] * n_plot

                self.plot_reconstructions(
                    X_cv[self.plot_ix],
                    (y_cv[self.plot_ix] * y_mask_cv[self.plot_ix][..., None]) if self.normalize_data else y_cv[self.plot_ix],
                    reconst[self.plot_ix],
                    titles=titles,
                    target_means=y_means_cv[self.plot_ix] if self.residual_decoder else None
                )

                self.plot_label_histogram(labels_pred)
                homogeneity = homogeneity_score(labels_cv, labels_pred)
                completeness = completeness_score(labels_cv, labels_pred)
                v_measure = v_measure_score(labels_cv, labels_pred)

                if verbose:
                    sys.stderr.write('Labeling scores (predictions):\n')
                    sys.stderr.write('  Homogeneity:  %s\n' % homogeneity)
                    sys.stderr.write('  Completeness: %s\n' % completeness)
                    sys.stderr.write('  V-measure:    %s\n\n' % v_measure)


                if self.binary_classifier or not self.k:
                    if not self.k:
                        k = 2 ** self.emb_dim
                    else:
                        k = 2 ** self.k
                else:
                    k = self.k

                labels_rand = np.random.randint(0, k, labels_pred.shape)

                if verbose:
                    sys.stderr.write('Labeling scores (random uniform):\n')
                    sys.stderr.write('  Homogeneity:  %s\n' % homogeneity_score(labels_cv, labels_rand))
                    sys.stderr.write('  Completeness: %s\n' % completeness_score(labels_cv, labels_rand))
                    sys.stderr.write('  V-measure:    %s\n\n' % v_measure_score(labels_cv, labels_rand))

                self.set_predict_mode(False)

                return homogeneity, completeness, v_measure

    def fit(
            self,
            X,
            X_mask,
            y,
            y_mask,
            labels,
            X_cv=None,
            X_mask_cv=None,
            y_cv=None,
            y_mask_cv=None,
            labels_cv=None,
            n_iter=None,
            ix2label=None,
            n_plot=10
    ):
        usingGPU = tf.test.is_gpu_available()
        sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if X_cv is None or X_mask_cv is None or y_cv is None or y_mask_cv is None or labels_cv is None:
            if self.plot_ix is None or len(self.plot_ix) != n_plot:
                self.plot_ix = np.random.choice(np.arange(len(X)), size=n_plot)
            X_cv = X
            X_mask_cv = X_mask
            y_cv = y
            y_mask_cv = y_mask
            labels_cv = labels
        else:
            if self.plot_ix is None or len(self.plot_ix) != n_plot:
                self.plot_ix = np.random.choice(np.arange(len(X_cv)), size=n_plot)

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
            y_means = None
            y_means_cv = None

        if n_iter is None:
            n_iter = self.n_iter

        sys.stderr.write('*' * 100 + '\n')
        sys.stderr.write(self.report_settings())
        sys.stderr.write('*' * 100 + '\n\n')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not np.isfinite(self.minibatch_size):
                    minibatch_size = len(y)
                else:
                    minibatch_size = self.minibatch_size
                n_minibatch = math.ceil(float(len(y)) / minibatch_size)

                while self.global_step.eval(session=self.sess) < n_iter:
                    perm, perm_inv = get_random_permutation(len(y))
                    t0_iter = time.time()
                    sys.stderr.write('-' * 50 + '\n')
                    sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                    sys.stderr.write('\n')
                    if self.optim_name is not None and self.lr_decay_family is not None:
                        sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                    pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                    loss_total = 0.

                    for j in range(0, len(y), self.minibatch_size):
                        indices = perm[j:j+minibatch_size]
                        fd_minibatch = {
                            self.X: X[indices],
                            self.X_mask: X_mask[indices],
                            self.y: y[indices],
                            self.y_mask: y_mask[indices],
                        }

                        # variances = self.sess.run(self.decoder_scale, feed_dict=fd_minibatch)
                        # print(variances)
                        # print(variances.min())
                        # print(variances.max())
                        # print()

                        info_dict = self.run_train_step(fd_minibatch)
                        metric_cur = info_dict['loss']

                        self.sess.run(self.ema_op)
                        if not np.isfinite(metric_cur):
                            metric_cur = 0
                        loss_total += metric_cur

                        self.sess.run(self.incr_global_batch_step)
                        pb.update((j/minibatch_size)+1, values=[('loss', metric_cur)])

                    loss_total /= n_minibatch

                    self.sess.run(self.incr_global_step)

                    if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        try:
                            self.check_numerics()
                            numerics_passed = True
                        except:
                            numerics_passed = False

                        if numerics_passed:
                            sys.stderr.write('Saving model...\n')

                            self.save()

                            homogeneity, completeness, v_measure = self.run_incremental_evaluation(
                                X_cv,
                                X_mask_cv,
                                y_cv,
                                y_mask_cv,
                                labels_cv,
                                ix2label=ix2label,
                                y_means_cv=None if y_means_cv is None else y_means_cv
                            )

                            fd_summary = {
                                self.loss_summary: loss_total,
                                self.homogeneity: homogeneity,
                                self.completeness: completeness,
                                self.v_measure: v_measure
                            }
                            summary_metrics = self.sess.run(self.summary_metrics, feed_dict=fd_summary)
                            self.writer.add_summary(summary_metrics, self.global_step.eval(session=self.sess))

                        else:
                            sys.stderr.write('Numerics check failed. Aborting save and reloading from previous checkpoint...\n')
                            self.load()

                    t1_iter = time.time()
                    sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

    def plot_reconstructions(self, inputs, targets, preds, titles=None, target_means=None, dir=None):
        if dir is None:
            dir = self.outdir

        plot_acoustic_features(
            inputs,
            targets,
            preds,
            titles=titles,
            target_means=target_means,
            dir=dir
        )

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
            dir=dir
        )


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

    def load(self, dir=None, predict=False, restore=True):
        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if restore and os.path.exists(dir + '/checkpoint'):
                    try:
                        if predict:
                            self.ema_saver.restore(self.sess, dir + '/model.ckpt')
                        else:
                            self.saver.restore(self.sess, dir + '/model.ckpt')
                    except:
                        if predict:
                            self.ema_saver.restore(self.sess, dir + '/model_backup.ckpt')
                        else:
                            self.saver.restore(self.sess, dir + '/model_backup.ckpt')
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')
                    self.sess.run(tf.global_variables_initializer())

    def set_predict_mode(self, mode):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.sess.run(self.training_on if mode else self.training_off)
                self.load(predict=mode)

    def report_settings(self, indent=0):
        out = ' ' * indent + 'MODEL SETTINGS:\n'
        out += ' ' * (indent + 2) + 'k: %s\n' %self.k
        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out

class UnsupervisedWordClassifierMLE(UnsupervisedWordClassifier):
    _INITIALIZATION_KWARGS = UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS

    _doc_header = """
        MLE implementation of unsupervised word classifier.

    """
    _doc_args = UnsupervisedWordClassifier._doc_args
    _doc_kwargs = UnsupervisedWordClassifier._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value,
                                                                                                  str) else "'%s'" % x.default_value)
                                     for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    def __init__(self, k, **kwargs):
        super(UnsupervisedWordClassifierMLE, self).__init__(
            k=k,
            **kwargs
        )

        for kwarg in UnsupervisedWordClassifierMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        kwarg_keys = [x.key for x in UnsupervisedWordClassifier._INITIALIZATION_KWARGS]
        for kwarg_key in kwargs:
            if kwarg_key not in kwarg_keys:
                raise TypeError('__init__() got an unexpected keyword argument %s' %kwarg_key)

        self._initialize_metadata()

    def _initialize_metadata(self):
        super(UnsupervisedWordClassifierMLE, self)._initialize_metadata()

    def _pack_metadata(self):
        md = super(UnsupervisedWordClassifierMLE, self)._pack_metadata()

        for kwarg in UnsupervisedWordClassifierMLE._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        super(UnsupervisedWordClassifierMLE, self)._unpack_metadata(md)

        for kwarg in UnsupervisedWordClassifierMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            sys.stderr.write('Saved model contained unrecognized attributes %s which are being ignored\n' %sorted(list(md.keys())))

    def _initialize_classifier(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.binary_classifier:
                    self.encoding = tf.sigmoid(self.encoder[:, :self.k])
                else:
                    self.encoding = tf.nn.softmax(self.encoder[:, :self.k])

    def _initialize_output_model(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.normalize_data and self.constrain_output:
                    self.out = tf.sigmoid(self.decoder)
                else:
                    self.out = self.decoder

    def _initialize_objective(self, n_train):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                # Define access points to important layers
                self.reconst = self.out
                self.reconst *= self.y_mask[..., None]

                self.encoding_post = self.encoding
                self.labels_post = self.labels
                self.label_probs_post = self.label_probs

                if self.normalize_data and self.constrain_output:
                    if self.mask_padding:
                        loss = tf.losses.log_loss(self.y, self.out, weights=self.y_mask[..., None])
                    else:
                        loss = tf.losses.log_loss(self.y, self.out)
                else:
                    if self.mask_padding:
                        loss = tf.losses.mean_squared_error(self.y, self.out, weights=self.y_mask[..., None])
                    else:
                        loss = tf.losses.mean_squared_error(self.y, self.out)

                self.loss = loss
                self.optim = self._initialize_optimizer(self.optim_name)
                self.train_op = self.optim.minimize(self.loss, global_step=self.global_batch_step)

    def run_train_step(self, feed_dict, return_loss=True, return_reconstructions=False, return_labels=False, return_label_probs=False):
        out_dict = {}

        if return_loss or return_reconstructions or return_labels or return_label_probs:
            to_run = [self.train_op]
            to_run_names = []
            if return_loss:
                to_run.append(self.loss)
                to_run_names.append('loss')
            if return_reconstructions:
                to_run.append(self.reconst)
                to_run_names.append('reconst')
            if return_labels:
                to_run.append(self.labels)
                to_run_names.append('labels')
            if return_label_probs:
                to_run.append(self.label_probs)
                to_run_names.append('label_probs')

            output = self.sess.run(to_run, feed_dict=feed_dict)
            for i, x in enumerate(output[1:]):
                out_dict[to_run_names[i]] = x

        return out_dict

    def report_settings(self, indent=0):
        out = super(UnsupervisedWordClassifierMLE, self).report_settings(indent=indent)
        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out





