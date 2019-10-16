import sys
import os
import tensorflow as tf
import edward as ed
from edward.models import OneHotCategorical, RelaxedOneHotCategorical, Bernoulli, RelaxedBernoulli, Normal, MultivariateNormalTriL, TransformedDistribution

from .kwargs import UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS
from .model import DenseLayer, DenseResidualLayer, DNNSeg, get_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


def bernoulli_rv(probs, inference_map, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            out = Bernoulli(probs=tf.ones_like(probs) * 00.5)
            out_post = Bernoulli(probs=probs)
            inference_map[out] = out_post
            
            return (out, inference_map)


class DNNSegBayes(DNNSeg):
    _INITIALIZATION_KWARGS = UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS

    _doc_header = """
           Bayesian implementation of unsupervised word classifier.

       """
    _doc_args = DNNSeg._doc_args
    _doc_kwargs = DNNSeg._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value,
                                                                                                  str) else "'%s'" % x.default_value)
                                     for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    def __init__(self, k, train_data, **kwargs):
        super(DNNSegBayes, self).__init__(
            k,
            train_data,
            **kwargs
        )

        for kwarg in DNNSegBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        kwarg_keys = [x.key for x in DNNSeg._INITIALIZATION_KWARGS]
        for kwarg_key in kwargs:
            if kwarg_key not in kwarg_keys:
                raise TypeError('__init__() got an unexpected keyword argument %s' % kwarg_key)

        assert self.declare_priors or self.relaxed, 'Priors must be explicitly declared unless relaxed==True'

        self._initialize_metadata()

    def _initialize_metadata(self):
        super(DNNSegBayes, self)._initialize_metadata()

        self.inference_map = {}

    def _pack_metadata(self):
        md = super(DNNSegBayes, self)._pack_metadata()

        for kwarg in DNNSegBayes._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        super(DNNSegBayes, self)._unpack_metadata(md)

        for kwarg in DNNSegBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            sys.stderr.write(
                'Saved model contained unrecognized attributes %s which are being ignored\n' % sorted(list(md.keys())))

    def _initialize_classifier(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.trainable_temp:
                    temp = tf.Variable(self.temp, name='temp')
                else:
                    temp = self.temp
                if self.binary_classifier:
                    if self.relaxed:
                        self.encoding_q = RelaxedBernoulli(temp, logits = self.encoder[:,:self.k])
                    else:
                        self.encoding_q = Bernoulli(logits = self.encoder[:,:self.k], dtype=self.FLOAT_TF)
                    if self.declare_priors:
                        if self.relaxed:
                            self.encoding = RelaxedBernoulli(temp, probs =tf.ones([tf.shape(self.y_bwd)[0], self.k]) * 0.5)
                        else:
                            self.encoding = Bernoulli(probs =tf.ones([tf.shape(self.y_bwd)[0], self.k]) * 0.5, dtype=self.FLOAT_TF)
                        if self.k:
                            self.inference_map[self.encoding] = self.encoding_q
                    else:
                        self.encoding = self.encoding_q
                else:
                    if self.relaxed:
                        self.encoding_q = RelaxedOneHotCategorical(temp, logits = self.encoder[:,:self.k])
                    else:
                        self.encoding_q = OneHotCategorical(logits = self.encoder[:,:self.k], dtype=self.FLOAT_TF)
                    if self.declare_priors:
                        if self.relaxed:
                            self.encoding = RelaxedOneHotCategorical(temp, probs =tf.ones([tf.shape(self.y_bwd)[0], self.k]) / self.k)
                        else:
                            self.encoding = OneHotCategorical(probs =tf.ones([tf.shape(self.y_bwd)[0], self.k]) / self.k, dtype=self.FLOAT_TF)
                        if self.k:
                            self.inference_map[self.encoding] = self.encoding_q
                    else:
                        self.encoding = self.encoding_q

    def _initialize_decoder_scale(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                dim = self.n_timesteps_output_bwd * self.frame_dim

                if self.decoder_type == 'rnn':

                    decoder_scale = tf.nn.softplus(
                        tf.keras.layers.LSTM(
                            self.frame_dim,
                            recurrent_activation='sigmoid',
                            return_sequences=True,
                            unroll=self.unroll
                        )(self.decoder_in)
                    )

                elif self.decoder_type == 'cnn':
                    assert self.n_timesteps_output_bwd is not None, 'n_timesteps_output must be defined when decoder_type == "cnn"'

                    decoder_scale = tf.layers.dense(self.decoder_in, self.n_timesteps * self.frame_dim)[..., None]
                    decoder_scale = tf.reshape(decoder_scale, (self.batch_len, self.n_timesteps_output_bwd, self.frame_dim, 1))
                    decoder_scale = tf.keras.layers.Conv2D(self.conv_n_filters, self.conv_kernel_size, padding='same', activation='elu')(decoder_scale)
                    decoder_scale = tf.keras.layers.Conv2D(1, self.conv_kernel_size, padding='same', activation='linear')(decoder_scale)
                    decoder_scale = tf.squeeze(decoder_scale, axis=-1)

                elif self.decoder_type in ['dense', 'dense_resnet']:
                    n_classes = int(2 ** self.k) if self.binary_classifier else int(self.k)

                    # First layer
                    decoder_scale = DenseLayer(
                        self.decoder_in,
                        self.n_timesteps_output_bwd * self.frame_dim,
                        self.training,
                        activation=tf.nn.elu,
                        batch_normalize=self.batch_normalize,
                        session=self.sess
                    )

                    # Intermediate layers
                    if self.decoder_type == 'dense':
                        for i in range(1, self.n_layers_decoder - 1):
                            decoder_scale = DenseLayer(
                                decoder_scale,
                                self.n_timesteps_output_bwd * self.frame_dim,
                                self.training,
                                activation=tf.nn.elu,
                                batch_normalize=self.batch_normalize,
                                session=self.sess
                            )
                    else: # self.decoder_type = 'dense_resnet'
                        for i in range(1, self.n_layers_decoder - 1):
                            decoder_scale = DenseResidualLayer(
                                decoder_scale,
                                self.training,
                                units=self.n_timesteps_output_bwd * self.frame_dim,
                                layers_inner=self.resnet_n_layers_inner,
                                activation_inner=tf.nn.elu,
                                activation=None,
                                batch_normalize=self.batch_normalize,
                                session=self.sess
                            )

                    # Last layer
                    if self.n_layers_decoder > 1:
                        decoder_scale = DenseLayer(
                            decoder_scale,
                            self.n_timesteps_output_bwd * self.frame_dim,
                            self.training,
                            activation=None,
                            batch_normalize=False,
                            session=self.sess
                        )

                    # Reshape
                    decoder_scale = tf.reshape(decoder_scale, (self.batch_len, self.n_timesteps_output_bwd, self.frame_dim))

                else:
                    raise ValueError('Decoder type "%s" not supported at this time' %self.decoder_type)

                self.decoder_scale = tf.nn.softplus(decoder_scale) + self.epsilon

    # Override this method to include scale params for output distribution
    def _initialize_decoder(self):
        super(DNNSegBayes, self)._initialize_decoder()
        self._initialize_decoder_scale()

    def _initialize_output_model(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.output_scale is None:
                    output_scale = self.decoder_scale
                else:
                    output_scale = self.output_scale

                if self.mv:
                    self.out = MultivariateNormalTriL(
                        loc=tf.layers.Flatten()(self.decoder),
                        scale_tril= output_scale
                    )
                else:
                    self.out = Normal(
                        loc=self.decoder,
                        scale = output_scale
                    )
                if self.normalize_data and self.constrain_output:
                    self.out = TransformedDistribution(
                        self.out,
                        bijector=tf.contrib.distributions.bijectors.Sigmoid()
                    )


    def _initialize_objective(self, n_train):
        n_train_minibatch = self.n_minibatch(n_train)
        minibatch_scale = self.minibatch_scale(n_train)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.mv:
                    y = tf.layers.Flatten()(self.y_bwd)
                    y_mask = tf.layers.Flatten()(self.y_bwd_mask[..., None] * tf.ones_like(self.y_bwd))
                else:
                    y = self.y_bwd
                    y_mask = self.y_bwd_mask

                # Define access points to important layers
                if len(self.inference_map) > 0:
                    self.out_post = ed.copy(self.out, self.inference_map)
                else:
                    self.out_post = self.out
                if self.mv:
                    self.reconst = tf.reshape(self.out_post * y_mask, [-1, self.n_timesteps_output_bwd, self.frame_dim])
                    # self.reconst_mean = tf.reshape(self.out_post.mean() * y_mask, [-1, self.n_timesteps_output, self.frame_dim])
                else:
                    self.reconst = self.out_post * y_mask[..., None]
                    # self.reconst_mean = self.out_post.mean() * y_mask[..., None]

                if len(self.inference_map) > 0:
                    self.encoding_post = ed.copy(self.encoding, self.inference_map)
                    self.labels_post = ed.copy(self.labels, self.inference_map)
                    self.label_probs_post = ed.copy(self.label_probs, self.inference_map)
                else:
                    self.encoding_post = self.encoding
                    self.labels_post = self.labels
                    self.label_probs_post = self.label_probs

                self.llprior = self.out.log_prob(y)
                self.ll_post = self.out_post.log_prob(y)

                self.optim = self._initialize_optimizer(self.optim_name)
                if self.variational():
                    self.inference = getattr(ed, self.inference_name)(self.inference_map, data={self.out: y})
                    if self.mask_padding:
                        self.inference.initialize(
                            n_samples=self.n_samples,
                            n_iter=n_train_minibatch * self.n_iter,
                            # n_print=n_train_minibatch * self.log_freq,
                            n_print=0,
                            logdir=self.outdir + '/tensorboard/edward',
                            log_timestamp=False,
                            scale={self.out: y_mask[...,None] * minibatch_scale},
                            optimizer=self.optim
                        )
                    else:
                        self.inference.initialize(
                            n_samples=self.n_samples,
                            n_iter=n_train_minibatch * self.n_iter,
                            # n_print=n_train_minibatch * self.log_freq,
                            n_print=0,
                            logdir=self.outdir + '/tensorboard/edward',
                            log_timestamp=False,
                            scale={self.out: minibatch_scale},
                            optimizer=self.optim
                        )
                else:
                    raise ValueError('Only variational inferences are supported at this time')

    def set_output_scale(self, output_scale, trainable=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if trainable:
                    self.output_scale = tf.Variable(output_scale, dtype=self.FLOAT_TF)
                    self.sess.run(tf.variables_initializer(self.output_scale))
                else:
                    self.output_scale = tf.constant(output_scale, dtype=self.FLOAT_TF)


    def run_train_step(
            self,
            feed_dict,
            return_loss=True,
            return_reconstructions=False,
            return_labels=False,
            return_label_probs=False,
            return_encoding_entropy=False,
            return_segmentation_probs=False
    ):
        info_dict = self.inference.update(feed_dict)

        out_dict = {}
        if return_loss:
            out_dict['loss'] = info_dict['loss']

        if return_reconstructions or return_labels or return_label_probs:
            to_run = []
            to_run_names = []
            if return_reconstructions:
                to_run.append(self.out_post)
                to_run_names.append('reconst')
            if return_labels:
                to_run.append(self.labels_post)
                to_run_names.append('labels')
            if return_label_probs:
                to_run.append(self.label_probs_post)
                to_run_names.append('label_probs')
            if return_encoding_entropy:
                to_run.append(self.encoding_entropy_mean)
                to_run_names.append('encoding_entropy')
            if self.encoder_type.lower() == 'softhmlstm' and return_segmentation_probs:
                to_run.append(self.segmentation_probs)
                to_run_names.append('segmentation_probs')

            output = self.sess.run(to_run, feed_dict=feed_dict)
            for i, x in enumerate(output):
                out_dict[to_run_names[i]] = x

        return out_dict

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

    def report_settings(self, indent=0):
        out = super(DNNSegBayes, self).report_settings(indent=indent)
        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out
