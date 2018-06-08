import sys
import os
import math
import time
import pickle
import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import RelaxedOneHotCategorical, RelaxedBernoulli, Normal
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

from .data import get_random_permutation
from .plot import plot_acoustic_features, plot_label_histogram

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class UnsupervisedWordClassifier(object):
    def __init__(
            self,
            k,
            temp=1.,
            trainable_temp=False,
            binary_classifier=False,
            emb_dim=None,
            encoder_type='rnn',
            decoder_type='rnn',
            dense_n_layers=1,
            unroll_rnn=False,
            conv_n_filters=16,
            conv_kernel_size=3,
            output_scale=None,
            n_coef=13,
            order=None,
            reconstruct_deltas=False,
            n_timesteps=None,
            float_type='float32',
            int_type='int32',
            n_iter = 1000,
            n_samples = 2,
            minibatch_size=128,
            eval_minibatch_size=100000,
            inference='KLqp',
            optim='Adam',
            learning_rate=0.01,
            learning_rate_min=1e-4,
            lr_decay_family=None,
            lr_decay_steps=25,
            lr_decay_rate=0.,
            lr_decay_staircase=False,
            max_global_gradient_norm = None,
            init_sd=1,
            ema_decay=0.999,
            log_freq=1,
            save_freq=1
    ):

        self.k = k
        self.temp = temp
        self.trainable_temp = trainable_temp
        self.binary_classifier = binary_classifier
        self.emb_dim = emb_dim

        assert not (unroll_rnn and n_timesteps is None), 'Number of timesteps must be prespecified for an unrolled RNN'
        assert not (encoder_type == 'dense' and n_timesteps is None), 'Number of timesteps must be prespecified for a dense encoder'
        assert not (encoder_type == 'cnn' and n_timesteps is None), 'Number of timesteps must be prespecified for a CNN encoder'
        assert not (decoder_type == 'dense' and n_timesteps is None), 'Number of timesteps must be prespecified for a dense decoder'
        assert not (decoder_type == 'cnn' and n_timesteps is None), 'Number of timesteps must be prespecified for a CNN decoder'
        assert encoder_type in ['rnn', 'cnn', 'dense'], 'Currently supported encoder types are "rnn", "cnn", and "dense"'
        assert decoder_type in ['rnn', 'cnn', 'dense'], 'Currently supported decoder types are "rnn", "cnn", and "dense"'

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.dense_n_layers = dense_n_layers
        self.unroll_rnn = unroll_rnn
        self.conv_n_filters = conv_n_filters
        self.conv_kernel_size = conv_kernel_size
        self.output_scale = output_scale
        self.n_coef = n_coef
        if order is None:
            self.order = [1, 2]
        else:
            self.order = sorted(list(set(order)))
        self.reconstruct_deltas = reconstruct_deltas
        self.n_timesteps = n_timesteps
        self.float_type = float_type
        self.int_type = int_type
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.minibatch_size = minibatch_size
        self.eval_minibatch_size = eval_minibatch_size
        self.optim_name = optim
        self.inference_name = inference
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.lr_decay_family = lr_decay_family
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_staircase = lr_decay_staircase
        self.max_global_gradient_norm = max_global_gradient_norm
        self.init_sd = init_sd
        self.ema_decay = ema_decay
        self.log_freq = log_freq
        self.save_freq = save_freq

        self.plot_ix = None

        self._initialize_metadata()

        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)
        self.regularizer_losses = []
        self.inference_map = {}

    def pack_metadata(self):
        return {
            'k': self.k,
            'temp': self.temp,
            'binary_classifier': self.binary_classifier,
            'emb_dim': self.emb_dim,
            'encoder_type': self.encoder_type,
            'decoder_type': self.decoder_type,
            'dense_n_layers': self.dense_n_layers,
            'unroll_rnn': self.unroll_rnn,
            'conv_n_filters': self.conv_n_filters,
            'conv_kernel_size': self.conv_kernel_size,
            'output_scale': self.output_scale,
            'n_coef': self.n_coef,
            'order': self.order,
            'reconstruct_deltas': self.reconstruct_deltas,
            'n_timestamps': self.n_timesteps,
            'float_type': self.float_type,
            'int_type': self.int_type,
            'n_iter': self.n_iter,
            'n_samples': self.n_samples,
            'minibatch_size': self.minibatch_size,
            'eval_minibatch_size': self.eval_minibatch_size,
            'optim_name': self.optim_name,
            'inference_name': self.inference_name,
            'learning_rate': self.learning_rate,
            'learning_rate_min': self.learning_rate_min,
            'lr_decay_family': self.lr_decay_family,
            'lr_decay_steps': self.lr_decay_steps,
            'lr_decay_rate': self.lr_decay_rate,
            'lr_decay_staircase': self.lr_decay_staircase,
            'init_sd': self.init_sd,
            'ema_decay': self.ema_decay,
            'log_freq': self.log_freq,
            'save_freq': self.save_freq
        }

    def unpack_metadata(self, md):

        self.k = md['k']
        self.temp = md.get('temp', 1.)
        self.binary_classifier = md.get('binary_classifier', False)
        self.encoder_type = md.get('encoder_type', 'rnn')
        self.decoder_type = md.get('decoder_type', 'rnn')
        self.dense_n_layers = md.get('dense_n_layers', 1)
        self.emb_dim = md.get('emb_dim', None)
        self.unroll_rnn = md.get('unroll_rnn', False)
        self.conv_n_filters = md.get('conv_n_filters', 16)
        self.conv_kernel_size = md.get('conv_kernel_size', 3)
        self.output_scale = md.get('output_scale', None)
        self.n_coef = md.get('n_coef', 13)
        self.order = md.get('order', [1,2])
        self.reconstruct_deltas = md.get('reconstruct_deltas', False)
        self.n_timesteps = md.get('n_timestamps', None)
        self.float_type = md.get('float_type', 'float32')
        self.int_type = md.get('int_type', 'int32')
        self.n_iter = md.get('n_iter', 1000)
        self.n_samples = md.get('n_samples', 2)
        self.minibatch_size = md.get('minibatch_size', 128)
        self.eval_minibatch_size = md.get('eval_minibatch_size', 100000)
        self.optim_name = md.get('optim_name', 'Adam')
        self.inference_name = md.get('inference_name', 'KLqp')
        self.learning_rate = md.get('learning_rate', 0.001)
        self.learning_rate_min = md.get('learning_rate_min', 1e-4)
        self.lr_decay_family = md.get('lr_decay_family', None)
        self.lr_decay_steps = md.get('lr_decay_steps', 25)
        self.lr_decay_rate = md.get('lr_decay_rate', 0.)
        self.lr_decay_staircase = md.get('lr_decay_staircase', False)
        self.init_sd = md.get('init_sd', 1.)
        self.ema_decay = md.get('ema_decay', 0.999)
        self.log_freq = md.get('log_freq', 1)
        self.save_freq = md.get('save_freq', 1)

    def __getstate__(self):
        return self.pack_metadata()

    def __setstate__(self, state):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        self.unpack_metadata(state)
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
        self._initialize_decoder()
        self._initialize_output_model()
        self._initialize_objective(n_train)
        self._initialize_ema()
        self._initialize_saver()
        self.load(restore=restore)

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.X = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps, self.n_coef * (1 + len(self.order))))
                self.X_mask = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps))

                if self.reconstruct_deltas:
                    self.frame_dim = self.n_coef * (1 + len(self.order))
                else:
                    self.frame_dim = self.n_coef
                self.y = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps, self.frame_dim))

                self.y_mask = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps))

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

                if self.binary_classifier:
                    binary_matrix = (np.expand_dims(np.arange(2 ** self.k), -1) & (1 << np.arange(self.k))).astype(bool).astype(int).T
                    self.binary_matrix = tf.constant(binary_matrix, dtype=self.FLOAT_TF)

                self.batch_len = tf.shape(self.X)[0]

    def _initialize_encoder(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.k
                if self.emb_dim:
                    units += self.emb_dim

                if self.encoder_type == 'rnn':
                    self.encoder = tf.keras.layers.LSTM(
                        units,
                        recurrent_activation='sigmoid',
                        return_sequences=False,
                        unroll=self.unroll_rnn
                    )(self.X)

                elif self.encoder_type == 'cnn':
                    encoder = tf.keras.layers.Conv2D(self.conv_n_filters, self.conv_kernel_size, strides=2, padding='same', activation='elu')(self.X[...,None])
                    encoder = tf.keras.layers.Conv2D(self.conv_n_filters * 2, self.conv_kernel_size, strides=2, padding='same', activation='elu')(encoder)
                    encoder = tf.layers.dense(tf.layers.Flatten()(encoder), self.k)
                    self.encoder = encoder

                elif self.encoder_type == 'dense':
                    encoder = tf.layers.Flatten()(self.X)
                    for _ in range(self.dense_n_layers):
                        encoder = tf.layers.dense(encoder, units)
                    self.encoder = encoder

    def _initialize_classifier(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.trainable_temp:
                    temp = tf.Variable(self.temp)
                else:
                    temp = self.temp
                if self.binary_classifier:
                    self.label = RelaxedBernoulli(temp, probs = tf.ones([tf.shape(self.y)[0], self.k]) * 0.5)
                    self.label_q = RelaxedBernoulli(temp, logits = self.encoder[:,:self.k])
                else:
                    self.label = RelaxedOneHotCategorical(temp, probs = tf.ones([tf.shape(self.y)[0], self.k]) / self.k)
                    self.label_q = RelaxedOneHotCategorical(temp, logits = self.encoder[:,:self.k])

                if self.emb_dim:
                    self.emb = tf.sigmoid(self.encoder[:,self.k:])
                    self.encoding = tf.concat([self.label, self.emb], axis=1)
                else:
                    self.encoding = self.label

                self.inference_map[self.label] = self.label_q

    def _initialize_decoder(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.decoder_type == 'rnn':
                    decoder_in = tf.tile(
                        tf.expand_dims(self.encoding, 1),
                        [1, tf.shape(self.y)[1], 1]
                    ) * self.y_mask[...,None]

                    self.decoder = tf.keras.layers.LSTM(
                        self.y.shape[-1],
                        recurrent_activation='sigmoid',
                        return_sequences=True,
                        unroll=self.unroll_rnn
                    )(decoder_in)
                    self.decoder *= self.y_mask[...,None]

                    self.decoder_scale = tf.nn.softplus(
                        tf.keras.layers.LSTM(
                            self.frame_dim,
                            recurrent_activation='sigmoid',
                            return_sequences=True,
                            unroll=self.unroll_rnn
                        )(decoder_in)
                    )

                elif self.decoder_type == 'cnn':
                    assert self.n_timesteps is not None, 'n_timesteps must be defined when decoder_type == "cnn"'

                    # decoder = tf.layers.dense(self.encoding, self.n_timesteps * self.frame_dim)
                    # decoder = tf.reshape(decoder, (self.batch_len, self.n_timesteps, self.frame_dim, 1))
                    # decoder = tf.keras.layers.Conv2D(self.n_filters, self.kernel_size, padding='same', activation='elu')(decoder)
                    # decoder = tf.keras.layers.Conv2D(1, self.kernel_size, padding='same', activation='linear')(decoder)
                    # self.decoder = tf.squeeze(decoder, axis=-1)
                    #
                    # decoder_scale = tf.layers.dense(self.encoding, self.n_timesteps * self.frame_dim)
                    # decoder_scale = tf.reshape(decoder_scale, (self.batch_len, self.n_timesteps, self.frame_dim, 1))
                    # decoder_scale = tf.keras.layers.Conv2D(self.n_filters, self.kernel_size, padding='same', activation='elu')(decoder_scale)
                    # decoder_scale = tf.keras.layers.Conv2D(1, self.kernel_size, padding='same', activation='linear')(decoder_scale)
                    # self.decoder_scale = tf.nn.softplus(tf.squeeze(decoder_scale, axis=-1))

                    decoder = tf.layers.dense(self.encoding, self.n_timesteps)[..., None]
                    decoder = tf.keras.layers.Conv1D(self.frame_dim, self.conv_kernel_size, padding='same', activation='elu')(decoder)
                    decoder = tf.keras.layers.Conv2D(self.conv_n_filters, self.conv_kernel_size, padding='same', activation='linear')(decoder[..., None])
                    decoder = tf.keras.layers.Conv2D(1, self.conv_kernel_size, padding='same', activation='linear')(decoder)
                    decoder = tf.squeeze(decoder, axis=-1)
                    self.decoder = decoder

                    decoder_scale = tf.layers.dense(self.encoding, self.n_timesteps)[..., None]
                    decoder_scale = tf.keras.layers.Conv1D(self.frame_dim, self.conv_kernel_size, padding='same', activation='elu')(decoder_scale)
                    decoder_scale = tf.keras.layers.Conv2D(self.conv_n_filters, self.conv_kernel_size, padding='same', activation='linear')(decoder_scale[..., None])
                    decoder_scale = tf.keras.layers.Conv2D(1, self.conv_kernel_size, padding='same', activation='linear')(decoder_scale)
                    decoder_scale = tf.squeeze(decoder_scale, axis=-1)
                    self.decoder_scale = decoder_scale

                elif self.decoder_type == 'dense':
                    decoder = self.encoding
                    decoder_scale = self.encoding
                    for _ in range(self.dense_n_layers):
                        decoder = tf.layers.dense(decoder, self.n_timesteps * self.frame_dim)
                        decoder_scale = tf.layers.dense(decoder_scale, self.n_timesteps * self.frame_dim)

                    self.decoder = tf.reshape(decoder, (self.batch_len, self.n_timesteps, self.frame_dim))
                    self.decoder_scale = tf.reshape(tf.nn.softplus(decoder_scale), (self.batch_len, self.n_timesteps, self.frame_dim))

    def _initialize_output_model(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.output_scale is None:
                    output_scale = self.decoder_scale
                else:
                    output_scale = self.output_scale
                self.out = Normal(
                    loc=self.decoder,
                    scale = output_scale
                )

    def _initialize_objective(self, n_train):
        n_train_minibatch = self.n_minibatch(n_train)
        minibatch_scale = self.minibatch_scale(n_train)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.optim = self._initialize_optimizer(self.optim_name)
                if self.variational():
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={self.out: self.y})
                    self.inference.initialize(
                        n_samples=self.n_samples,
                        n_iter=self.n_iter,
                        # n_print=n_train_minibatch * self.log_freq,
                        n_print=0,
                        logdir=self.outdir + '/tensorboard/edward',
                        log_timestamp=False,
                        scale={self.out: self.y_mask[...,None]},
                        # scale={self.out: minibatch_scale * self.y_mask[...,None]},
                        optimizer=self.optim
                    )
                else:
                    raise ValueError('Only variational inferences are supported at this time')

                self.out_post = ed.copy(self.out, self.inference_map)

                if self.binary_classifier:
                    label = tf.expand_dims(self.label, -1) * self.binary_matrix + (1 - tf.expand_dims(self.label, -1)) * (1 - self.binary_matrix)
                    label = tf.reduce_prod(label, -2)
                else:
                    label = self.label

                self.label_post = ed.copy(label, self.inference_map)

                label_hard = tf.argmax(label, axis=1)
                self.label_hard_post = ed.copy(label_hard, self.inference_map)

                self.llprior = self.out.log_prob(self.y)
                self.ll_post = self.out_post.log_prob(self.y)

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
                    'SGD': lambda x: self._clipped_optimizer_class(tf.train.GradientDescentOptimizer)(x, max_global_norm=clip),
                    'Momentum': lambda x: self._clipped_optimizer_class(tf.train.MomentumOptimizer)(x, 0.9, max_global_norm=clip),
                    'AdaGrad': lambda x: self._clipped_optimizer_class(tf.train.AdagradOptimizer)(x, max_global_norm=clip),
                    'AdaDelta': lambda x: self._clipped_optimizer_class(tf.train.AdadeltaOptimizer)(x, max_global_norm=clip),
                    'Adam': lambda x: self._clipped_optimizer_class(tf.train.AdamOptimizer)(x, max_global_norm=clip),
                    'FTRL': lambda x: self._clipped_optimizer_class(tf.train.FtrlOptimizer)(x, max_global_norm=clip),
                    'RMSProp': lambda x: self._clipped_optimizer_class(tf.train.RMSPropOptimizer)(x, max_global_norm=clip),
                    'Nadam': lambda x: self._clipped_optimizer_class(tf.contrib.opt.NadamOptimizer)(x, max_global_norm=clip)
                }[name](self.lr)

    ## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
    def _clipped_optimizer_class(self, base_optimizer):
        class ClippedOptimizer(base_optimizer):
            def __init__(self, *args, max_global_norm=None, **kwargs):
                super(ClippedOptimizer, self).__init__( *args, **kwargs)
                self.max_global_norm = max_global_norm

            def compute_gradients(self, *args, **kwargs):
                grad_and_vars = super(ClippedOptimizer, self).compute_gradients(*args, **kwargs)
                if self.max_global_norm is None:
                    return grad_and_vars
                return tf.clip_by_global_norm([g for g, _ in grad_and_vars], self.max_global_norm)

        return ClippedOptimizer


    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

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

    def n_minibatch(self, n):
        return math.ceil(float(n) / self.minibatch_size)

    def minibatch_scale(self, n):
        return float(n) / self.minibatch_size

    def variational(self):
        """
        Check whether the DTSR model uses variational Bayes.

        :return: ``True`` if the model is variational, ``False`` otherwise.
        """
        return self.inference_name in [
            'KLpq',
            'KLqp',
            'ImplicitKLqp',
            'ReparameterizationEntropyKLqp',
            'ReparameterizationKLKLqp',
            'ReparameterizationKLqp',
            'ScoreEntropyKLqp',
            'ScoreKLKLqp',
            'ScoreKLqp',
            'ScoreRBKLqp',
            'WakeSleep'
        ]

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

        if n_iter is None:
            n_iter = self.n_iter


        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not np.isfinite(self.minibatch_size):
                    minibatch_size = len(y)
                else:
                    minibatch_size = self.minibatch_size
                n_minibatch = math.ceil(float(len(y)) / minibatch_size)

                while self.global_step.eval(session=self.sess) < n_iter:
                    p, p_inv = get_random_permutation(len(y))
                    t0_iter = time.time()
                    sys.stderr.write('-' * 50 + '\n')
                    sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                    sys.stderr.write('\n')
                    if self.optim_name is not None and self.lr_decay_family is not None:
                        sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                    pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                    loss_total = 0.

                    for j in range(0, len(y), self.minibatch_size):

                        indices = p[j:j+minibatch_size]
                        fd_minibatch = {
                            self.X: X[indices],
                            self.X_mask: X_mask[indices],
                            self.y: y[indices],
                            self.y_mask: y_mask[indices],
                        }

                        info_dict = self.inference.update(fd_minibatch)
                        self.sess.run(self.ema_op)
                        metric_cur = info_dict['loss']
                        if not np.isfinite(metric_cur):
                            metric_cur = 0
                        loss_total += metric_cur

                        self.sess.run(self.incr_global_batch_step)
                        pb.update((j/minibatch_size)+1, values=[('loss', metric_cur)])

                    self.sess.run(self.incr_global_step)
                    if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        sys.stderr.write('Saving model...\n')
                        self.save()

                        self.set_predict_mode(True)

                        sys.stderr.write('Extracting predictions...\n')
                        _, reconst, labels_pred = self.sess.run(
                            [self.incr_global_step, self.out_post, self.label_hard_post],
                            feed_dict={
                                self.X: X_cv,
                                self.X_mask: X_mask_cv,
                                self.y: y_cv,
                                self.y_mask: y_mask_cv
                            }
                        )

                        sys.stderr.write('Plotting...\n')
                        self.plot_reconstructions(
                            X_cv[self.plot_ix],
                            y_cv[self.plot_ix],
                            reconst[self.plot_ix]
                        )

                        self.plot_label_histogram(labels_pred)

                        sys.stderr.write('Homogeneity: %s\n' %homogeneity_score(labels_cv, labels_pred))
                        sys.stderr.write('Completeness: %s\n' %completeness_score(labels_cv, labels_pred))
                        sys.stderr.write('V-measure: %s\n\n' %v_measure_score(labels_cv, labels_pred))

                        self.set_predict_mode(False)

                    t1_iter = time.time()
                    sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

    def plot_reconstructions(self, inputs, targets, preds, dir=None):
        if dir is None:
            dir = self.outdir

        plot_acoustic_features(
            inputs,
            targets,
            preds,
            dir=dir
        )

    def plot_label_histogram(self, labels_pred, dir=None):
        if dir is None:
            dir = self.outdir

        if self.binary_classifier:
            bins = 2 ** self.k
        else:
            bins = self.k

        plot_label_histogram(labels_pred, dir=dir, bins=bins)

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
                self.load(predict=mode)



