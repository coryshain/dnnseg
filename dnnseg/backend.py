import re
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers.utils import conv_output_length


if hasattr(rnn_cell_impl, 'LayerRNNCell'):
    LayerRNNCell = rnn_cell_impl.LayerRNNCell
else:
    LayerRNNCell = rnn_cell_impl._LayerRNNCell


parse_initializer = re.compile('(.*_initializer)(_(.*))?')


def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess


# Debugged reduce_logsumexp to allow -inf
# Stolen from meijun on Github <https://github.com/tensorflow/tensorflow/issues/11692>
def tf_reduce_logsumexp(input_tensor,
                        axis=None,
                        keep_dims=False,
                        name=None,
                        reduction_indices=None,
                        session=None):
    """Fix tf.reduce_logsumexp"""
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            with tf.name_scope(name, "tf_ReduceLogSumExp", [input_tensor]) as name:
                raw_max = tf.reduce_max(
                    input_tensor,
                    axis=axis,
                    reduction_indices=reduction_indices,
                    keep_dims=True)
                my_max = tf.stop_gradient(
                    tf.where(
                        tf.is_finite(raw_max),
                        raw_max,
                        tf.zeros_like(raw_max)))
                result = tf.log(
                    tf.reduce_sum(
                        tf.exp(input_tensor - my_max),
                        axis,
                        keep_dims=True,
                        reduction_indices=reduction_indices)) + my_max
                if not keep_dims:
                    if isinstance(axis, int):
                        axis = [axis]
                    result = tf.squeeze(result, axis)
                return result

def get_activation(activation, session=None, training=True, from_logits=True, sample_at_train=True, sample_at_eval=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            hard_sigmoid = tf.keras.backend.hard_sigmoid

            if activation:
                if isinstance(activation, str) and activation.lower() == 'hard_sigmoid':
                    out = hard_sigmoid
                elif isinstance(activation, str) and activation.lower() == 'bsn':
                    def make_sample_fn(s, from_logits):
                        if from_logits:
                            def sample_fn(x):
                                return bernoulli_straight_through(tf.sigmoid(x), session=s)
                        else:
                            def sample_fn(x):
                                return bernoulli_straight_through(x, session=s)

                        return sample_fn

                    def make_round_fn(s, from_logits):
                        if from_logits:
                            def round_fn(x):
                                return round_straight_through(tf.sigmoid(x), session=s)
                        else:
                            def round_fn(x):
                                return round_straight_through(x, session=s)

                        return round_fn

                    sample_fn = make_sample_fn(session, from_logits)
                    round_fn = make_round_fn(session, from_logits)

                    if sample_at_train:
                        train_fn = sample_fn
                    else:
                        train_fn = round_fn

                    if sample_at_eval:
                        eval_fn = sample_fn
                    else:
                        eval_fn = round_fn

                    out = lambda x: tf.cond(training, lambda: train_fn(x), lambda: eval_fn(x))

                elif isinstance(activation, str) and activation.lower().startswith('slow_sigmoid'):
                    split = activation.split('_')
                    if len(split) == 2:
                        # Default to a slowness parameter of 1/2
                        scale = 0.5
                    else:
                        try:
                            scale = float(split[2])
                        except ValueError:
                            raise ValueError('Parameter to slow_sigmoid must be a valid float.')

                    out = lambda x: tf.sigmoid(0.5 * x)

                elif isinstance(activation, str):
                    out = getattr(tf.nn, activation)
                else:
                    out = activation
            else:
                out = lambda x: x

    return out


def get_initializer(initializer, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if isinstance(initializer, str):
                initializer_name, _, initializer_params = parse_initializer.match(initializer).groups()

                kwargs = {}
                if initializer_params:
                    kwarg_list = initializer_params.split('-')
                    for kwarg in kwarg_list:
                        key, val = kwarg.split('=')
                        try:
                            val = float(val)
                        except Exception:
                            pass
                        kwargs[key] = val

                out = getattr(tf, initializer_name)
                if 'glorot' in initializer:
                    out = out()
                else:
                    out = out(**kwargs)
            else:
                out = initializer

            return out


def get_regularizer(init, scale=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if scale is None:
                scale = 0.001

            if init is None:
                out = None
            elif isinstance(init, str):
                out = getattr(tf.contrib.layers, init)(scale=scale)
            elif isinstance(init, float):
                out = tf.contrib.layers.l2_regularizer(scale=init)
            else:
                raise ValueError('Unrecognized value "%s" for init parameter of get_regularizer()' %init)

            return out


def get_dropout(rate, training=True, noise_shape=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if rate:
                def make_dropout(rate):
                    return lambda x: tf.layers.dropout(x, rate=rate, noise_shape=noise_shape, training=training)
                out = make_dropout(rate)
            else:
                out = lambda x: x

            return out


def round_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = tf.round
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)
            # with ops.name_scope("BinaryRound") as name:
            #     with session.graph.gradient_override_map({"Round": "Identity"}):
            #         return tf.round(x, name=name)


def bernoulli_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = lambda x: tf.ceil(x - tf.random_uniform(tf.shape(x)))
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)
            # with ops.name_scope("BernoulliSample") as name:
            #     with session.graph.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):
            #         return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)


@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]


def replace_gradient(fw_op, bw_op, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            def new_op(x):
                fw = fw_op(x)
                bw = bw_op(x)
                out = bw + tf.stop_gradient(fw-bw)
                return out
            return new_op


def initialize_embeddings(categories, dim, default=0., session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            categories = sorted(list(set(categories)))
            n_categories = len(categories)
            index_table = tf.contrib.lookup.index_table_from_tensor(
                tf.constant(categories),
                num_oov_buckets=1
            )
            embedding_matrix = tf.Variable(tf.fill([n_categories+1, dim], default))

            return index_table, embedding_matrix


def binary2integer(b, int_type='int32', session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            tf_int_type = getattr(tf, int_type)

            k = int(b.shape[-1])
            if int_type.endswith('128'):
                assert k <= int(int_type[-3:]) / 2, 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' %(b.shape[-1], int_type)
            else:
                assert k <= int(int_type[-2:]) / 2, 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' %(b.shape[-1], int_type)
            base2 = 2 ** tf.range(k-1, limit=-1, delta=-1, dtype=tf_int_type)
            while len(base2.shape) < len(b.shape):
                base2 = tf.expand_dims(base2, 0)

            return tf.reduce_sum(tf.cast(b, dtype=tf_int_type) * base2, axis=-1)


def bernoulli2categorical(b, int_type='int32', float_type='float32', session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            tf_float_type = getattr(tf, float_type)

            k = int(b.shape[-1])
            if int_type.endswith('128'):
                assert k <= int(int_type[-3:]), 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' % (b.shape[-1], int_type)
            else:
                assert k <= int(int_type[-2:]), 'The number of classes (2 ** %d) exceeds the capacity of the current integer encoding ("%s")' % (b.shape[-1], int_type)

            binary_matrix = tf.constant((np.expand_dims(np.arange(2 ** k), -1) & (1 << np.arange(k))).astype(bool).astype(int).T, dtype=tf_float_type)
            c = tf.expand_dims(b, -1) * binary_matrix + (1 - tf.expand_dims(b, -1)) * (1 - binary_matrix)
            c = tf.reduce_prod(c, -2)
            return c


def sigmoid_to_logits(p, epsilon=1e-8, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            out = tf.clip_by_value(p, epsilon, 1-epsilon)
            out = tf.log(out / (1 - out))

            return out


def binary_entropy(p, from_logits=False, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if not from_logits:
                p = sigmoid_to_logits(p, session=session)

            out = tf.nn.sigmoid_cross_entropy_with_logits(logits=p, labels=tf.sigmoid(p))

            return out


def binary_entropy_regularizer(scale, from_logits=False, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            return lambda bit_probs: tf.reduce_mean(
                binary_entropy(bit_probs, from_logits=from_logits, session=session)
            ) * scale


def cross_entropy_regularizer(scale, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            return lambda x: tf.nn.sigmoid_cross_entropy_with_logits(labels=x[0], logits=x[1]) * scale


def mse_regularizer(scale, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            return lambda offsets: tf.reduce_mean(offsets ** 2) * scale

def identity_regularizer(scale, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            return lambda x: tf.reduce_mean(x) * scale


def compose_lambdas(lambdas, **kwargs):
    def composed_lambdas(x, **kwargs):
        out = x
        for l in lambdas:
            out = l(out, **kwargs)
        return out

    return composed_lambdas


def make_lambda(layer, session=None, use_kwargs=False):
    session = get_session(session)

    with session.as_default():
        with session.graph.as_default():
            if use_kwargs:
                def apply_layer(x, **kwargs):
                    return layer(x, **kwargs)
            else:
                def apply_layer(x, **kwargs):
                    return layer(x)
            return apply_layer


def ema(x, decay=0.9, axis=-1, session=None):
    session = get_session(session)

    with session.as_default():
        with session.graph.as_default():
            if axis < 0:
                axis += len(x.shape)

            assert axis > 0 and axis < len(x.shape), 'Axis %d is out of bounds for input with rank %d' % len(x.shape)

            permutation_axes = list(range(len(x.shape)))
            permutation_axes[0] = axis
            permutation_axes[axis] = 0

            x = tf.transpose(x, permutation_axes)

            def get_ema_func(decay):
                return lambda a, x: decay * a + (1 - decay) * x

            fn = get_ema_func(decay)
            initializer = x[0, ...]

            out = tf.scan(
                fn,
                x,
                initializer=initializer,
                swap_memory=True,
                parallel_iterations=32
            )

            out = tf.transpose(out, permutation_axes)

            return out


def dema(x, decay=0.9, axis=-1, method='brown', session=None):
    assert method.lower() in ['brown', 'holt-winters']
    session = get_session(session)


    with session.as_default():
        with session.graph.as_default():
            if axis < 0:
                axis += len(x.shape)

            assert axis > 0 and axis < len(x.shape), 'Axis %d is out of bounds for input with rank %d' % len(x.shape)

            permutation_axes = list(range(len(x.shape)))
            permutation_axes[0] = axis
            permutation_axes[axis] = 0

            x = tf.transpose(x, permutation_axes)

            initializer = x[0, ...]

            if method.lower() == 'brown':
                def get_ema_func(decay):
                    def double_ema(a, x):
                        s_tm1, sp_tm1 = a[0], a[1]
                        s_t = decay * s_tm1 + (1 - decay) * x
                        sp_t = decay * sp_tm1 + (1 - decay) * s_tm1
                        a_t = 2 * s_t - sp_t

                        out = tf.stack([s_t, sp_t, a_t], axis=0)

                        return out

                    return double_ema

                initializer = tf.stack([initializer, initializer, initializer], axis=0)

            else: # method.lower() == 'holt-winters'
                def get_ema_func(decay):
                    def double_ema(a, x):
                        b_tm1, s_tm1 = a[0], a[1]
                        s_t = decay * (s_tm1 + b_tm1) + (1 - decay) * x
                        b_t = decay * b_tm1 + (1 - decay) * (s_t - s_tm1)

                        out = tf.stack([b_t, s_t], axis=0)

                        return out

                    return double_ema

                initializer = tf.stack([initializer, tf.zeros_like(initializer)], axis=0)

            fn = get_ema_func(decay)

            out = tf.scan(
                fn,
                x,
                initializer=initializer,
                swap_memory=True,
                parallel_iterations=32
            )

            out = out[:, -1]

            out = tf.transpose(out, permutation_axes)

            return out


def wma(x, filter_width=5, session=None):
    with session.as_default():
        with session.graph.as_default():
            filter_left_width = tf.ceil(filter_width / 2)
            filter_right_width = tf.floor(filter_width / 2)
            pad_left_width = tf.cast(filter_left_width - 1, dtype=tf.int32)
            pad_right_width = tf.cast(filter_right_width, dtype=tf.int32)

            filter_left = tf.range(1, filter_left_width+1, dtype=tf.float32)
            filter_right = tf.range(1, filter_right_width+1, dtype=tf.float32)
            filter = tf.concat([filter_left, filter_right], axis=0)
            filter /= tf.reduce_sum(filter)
            filter = filter[..., None, None]

            x_pad_left = x[...,:1]
            tile_left_shape = tf.convert_to_tensor([1] * (len(x.shape) - 1) + [pad_left_width])

            x_pad_left = tf.tile(x_pad_left, tile_left_shape)
            x_pad_right = x[...,-1:]
            tile_right_shape = tf.convert_to_tensor([1] * (len(x.shape) - 1) + [pad_right_width])
            x_pad_right = tf.tile(x_pad_right, tile_right_shape)

            x = tf.concat([x_pad_left, x, x_pad_right], axis=-1)
            x = x[..., None]

            out = tf.nn.convolution(
                x,
                filter,
                'VALID',
                data_format='NWC'
            )
            out = out[..., -1]

            return out


class HMLSTMCell(LayerRNNCell):
    def __init__(
            self,
            num_units,
            num_layers,
            training=False,
            one_hot_inputs=False,
            forget_bias=1.0,
            oracle_boundary=False,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            boundary_activation='sigmoid',
            boundary_discretizer=None,
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            topdown_initializer='glorot_uniform_initializer',
            boundary_initializer='orthogonal_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            topdown_regularizer=None,
            boundary_regularizer=None,
            bias_regularizer=None,
            temporal_dropout=None,
            temporal_dropout_plug_lm=False,
            return_lm_loss=False,
            bottomup_dropout=None,
            recurrent_dropout=None,
            topdown_dropout=None,
            boundary_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            refeed_boundary=False,
            power=None,
            boundary_slope_annealing_rate=None,
            state_slope_annealing_rate=None,
            slope_annealing_max=None,
            state_discretizer=None,
            sample_at_train=True,
            sample_at_eval=False,
            global_step=None,
            implementation=1,
            reuse=None,
            name=None,
            dtype=None,
            session=None,
            device=None
    ):
        self._session = get_session(session)
        self._device = device

        with self._session.as_default():
            with self._session.graph.as_default():
                with tf.device(self._device):
                    super(HMLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                    if not isinstance(num_units, list):
                        self._num_units = [num_units] * num_layers
                    else:
                        self._num_units = num_units

                    assert len(
                        self._num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                    self._num_layers = num_layers

                    self._training = training

                    self._lm = return_lm_loss or temporal_dropout_plug_lm
                    self._return_lm_loss = return_lm_loss
                    self._temporal_dropout_plug_lm = temporal_dropout_plug_lm
                    self._one_hot_inputs = one_hot_inputs

                    self._forget_bias = forget_bias

                    self._oracle_boundary = oracle_boundary

                    self._activation = get_activation(activation, session=self._session, training=self._training)
                    self._inner_activation = get_activation(inner_activation, session=self._session, training=self._training)
                    assert not self._lm or inner_activation in ['tanh', tf.tanh], 'lm=True requires tanh inner activation.'
                    self._recurrent_activation = get_activation(recurrent_activation, session=self._session, training=self._training)
                    self._boundary_activation = get_activation(boundary_activation, session=self._session, training=self._training)
                    if boundary_discretizer:
                        self._boundary_discretizer = get_activation(
                            boundary_discretizer,
                            session=self._session,
                            training=self._training,
                            from_logits=False,
                            sample_at_train=sample_at_train,
                            sample_at_eval=sample_at_eval
                        )
                    else:
                        self._boundary_discretizer = None

                    self._bottomup_initializer = get_initializer(bottomup_initializer, session=self._session)
                    self._recurrent_initializer = get_initializer(recurrent_initializer, session=self._session)
                    self._topdown_initializer = get_initializer(topdown_initializer, session=self._session)
                    self._boundary_initializer = get_initializer(boundary_initializer, session=self._session)
                    self._bias_initializer = get_initializer(bias_initializer, session=self._session)

                    self._bottomup_regularizer = get_regularizer(bottomup_regularizer, session=self._session)
                    self._recurrent_regularizer = get_regularizer(recurrent_regularizer, session=self._session)
                    self._topdown_regularizer = get_regularizer(topdown_regularizer, session=self._session)
                    self._boundary_regularizer = get_regularizer(boundary_regularizer, session=self._session)
                    self._bias_regularizer = get_regularizer(bias_regularizer, session=self._session)

                    self._temporal_dropout = temporal_dropout
                    if isinstance(temporal_dropout, str):
                        temporal_dropout_tmp = []
                        for x in temporal_dropout.split():
                            try:
                                x = float(x)
                            except TypeError:
                                if isinstance(x, str):
                                    assert x.lower() == 'entropy', 'Unrecognized value for temporal dropout: "%s"' % x
                            temporal_dropout_tmp.append(x)
                        temporal_dropout = temporal_dropout_tmp
                        if len(temporal_dropout) == 1:
                            temporal_dropout = temporal_dropout * (self._num_layers)
                    elif not isinstance(temporal_dropout, list):
                        temporal_dropout = [temporal_dropout] * (self._num_layers)
                    else:
                        temporal_dropout = self.temporal_dropout_rate
                    assert len(temporal_dropout) == self._num_layers, 'Parameter temporal_dropout must be scalar or list of length ``num_layers``'
                    self._temporal_dropout = temporal_dropout

                    self._bottomup_dropout = get_dropout(bottomup_dropout, training=self._training, session=self._session)
                    self._recurrent_dropout = get_dropout(recurrent_dropout, training=self._training, session=self._session)
                    self._topdown_dropout = get_dropout(topdown_dropout, training=self._training, session=self._session)
                    self._boundary_dropout = get_dropout(boundary_dropout, training=self._training, session=self._session)

                    self.regularizer_losses = []

                    self._weight_normalization = weight_normalization
                    self._layer_normalization = layer_normalization
                    self._refeed_boundary = refeed_boundary
                    self._power = power
                    self._boundary_slope_annealing_rate = boundary_slope_annealing_rate
                    self._state_slope_annealing_rate = state_slope_annealing_rate
                    self._slope_annealing_max = slope_annealing_max
                    if state_discretizer:
                        self._state_discretizer = get_activation(
                            state_discretizer,
                            session=self._session,
                            training=self._training,
                            from_logits=False,
                            sample_at_train=sample_at_train,
                            sample_at_eval=sample_at_eval
                        )
                    else:
                        self._state_discretizer = None
                    self._sample_at_train = sample_at_train
                    self._sample_at_eval = sample_at_eval
                    self.global_step = global_step
                    self._implementation = implementation

                    self._epsilon = 1e-8

    def _regularize(self, var, regularizer):
        if regularizer is not None:
            with self._session.as_default():
                with self._session.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)

    def get_regularizer_losses(self):
        return self.regularizer_losses[:]

    @property
    def state_size(self):
        out = []
        for l in range(self._num_layers):
            if l < self._num_layers - 1:
                size = (self._num_units[l], self._num_units[l], 1)
            else:
                size = (self._num_units[l], self._num_units[l])

            out.append(size)

        out = tuple(out)

        return out

    @property
    def output_size(self):
        out = []
        for l in range(self._num_layers):
            if l < self._num_layers - 1:
                if self._return_lm_loss:
                    size = (self._num_units[l], 1, 1, 1)
                else:
                    size = (self._num_units[l], 1, 1)
            else:
                if self._return_lm_loss:
                    size = (self._num_units[l], 1)
                else:
                    size = (self._num_units[l],)

            out.append(size)

        out = tuple(out)

        return out

    def norm(self, inputs, name):
        with self._session.as_default():
            with self._session.graph.as_default():
                out = tf.contrib.layers.layer_norm(
                    inputs,
                    reuse=tf.AUTO_REUSE,
                    scope=name,
                    begin_norm_axis=-1,
                    begin_params_axis=-1
                )
                cond = tf.norm(inputs, axis=-1) > 0.
                out = tf.where(
                    cond,
                    out,
                    inputs
                )

                return out

    def build(self, inputs_shape):
        with self._session.as_default():
            with self._session.graph.as_default():
                with tf.device(self._device):
                    if inputs_shape[1].value is None:
                        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                         % inputs_shape)

                    self._input_dims = inputs_shape[1].value

                    self._kernel_bottomup = []
                    self._kernel_recurrent = []
                    self._kernel_topdown = []
                    self._bias_boundary = []
                    if self._lm:
                        self._kernel_lm_recurrent = []
                        self._kernel_lm_topdown = []
                        self._bias_lm = []

                    self._bias = []

                    if self._implementation == 2:
                        self._kernel_boundary = []

                    if self._oracle_boundary:
                        n_boundary_dims = self._num_layers - 1
                    else:
                        n_boundary_dims = 0

                    for l in range(self._num_layers):
                        if l == 0:
                            bottom_up_dim = inputs_shape[1].value - n_boundary_dims
                        else:
                            bottom_up_dim = self._num_units[l - 1]
                        bottom_up_dim += self._refeed_boundary and (self._implementation == 1)

                        recurrent_dim = self._num_units[l]

                        if self._implementation == 1:
                            if l < self._num_layers - 1:
                                output_dim = 4 * self._num_units[l] + 1
                            else:
                                output_dim = 4 * self._num_units[l]
                        else:
                            output_dim = 4 * self._num_units[l]

                        kernel_bottomup = self.add_variable(
                            'kernel_bottomup_%d' % l,
                            shape=[bottom_up_dim, output_dim],
                            initializer=self._bottomup_initializer
                        )
                        if self._weight_normalization:
                            kernel_bottomup_g = tf.Variable(tf.ones([1, output_dim]), name='kernel_bottomup_g_%d' % l)
                            kernel_bottomup_b = tf.Variable(tf.zeros([1, output_dim]), name='kernel_bottomup_b_%d' % l)
                            kernel_bottomup = tf.nn.l2_normalize(kernel_bottomup, axis=-1) * kernel_bottomup_g + kernel_bottomup_b
                        self._regularize(kernel_bottomup, self._bottomup_regularizer)
                        self._kernel_bottomup.append(kernel_bottomup)

                        kernel_recurrent = self.add_variable(
                            'kernel_recurrent_%d' % l,
                            shape=[recurrent_dim, output_dim],
                            initializer=self._recurrent_initializer
                        )
                        if self._weight_normalization:
                            kernel_recurrent_g = tf.Variable(tf.ones([1, output_dim]), name='kernel_recurrent_g_%d' % l)
                            kernel_recurrent_b = tf.Variable(tf.zeros([1, output_dim]), name='kernel_recurrent_b_%d' % l)
                            kernel_recurrent = tf.nn.l2_normalize(kernel_recurrent, axis=-1) * kernel_recurrent_g + kernel_recurrent_b
                        self._regularize(kernel_recurrent, self._recurrent_regularizer)
                        self._kernel_recurrent.append(kernel_recurrent)

                        if l < self._num_layers - 1:
                            top_down_dim = self._num_units[l + 1]
                            kernel_topdown = self.add_variable(
                                'kernel_topdown_%d' % l,
                                shape=[top_down_dim, output_dim],
                                initializer=self._topdown_initializer
                            )
                            if self._weight_normalization:
                                kernel_topdown_g = tf.Variable(tf.ones([1, output_dim]), name='kernel_topdown_g_%d' % l)
                                kernel_topdown_b = tf.Variable(tf.zeros([1, output_dim]), name='kernel_topdown_b_%s' % l)
                                kernel_topdown = tf.nn.l2_normalize(kernel_topdown, axis=-1) * kernel_topdown_g + kernel_topdown_b
                            self._regularize(kernel_topdown, self._topdown_regularizer)
                            self._kernel_topdown.append(kernel_topdown)

                        if self._lm:
                            lm_recurrent_in_dim = 2 * recurrent_dim
                            kernel_lm_recurrent = self.add_variable(
                                'kernel_lm_recurrent_%d' % l,
                                shape=[lm_recurrent_in_dim, bottom_up_dim],
                                initializer=self._recurrent_initializer
                            )
                            if self._weight_normalization:
                                kernel_lm_recurrent_g = tf.Variable(tf.ones([1, bottom_up_dim]), name='kernel_lm_recurrent_g_%d' % l)
                                kernel_lm_recurrent_b = tf.Variable(tf.zeros([1, bottom_up_dim]), name='kernel_lm_recurrent_b_%d' % l)
                                kernel_lm_recurrent = tf.nn.l2_normalize(kernel_lm_recurrent, axis=-1) * kernel_lm_recurrent_g + kernel_lm_recurrent_b
                            self._regularize(kernel_lm_recurrent, self._recurrent_regularizer)
                            self._kernel_lm_recurrent.append(kernel_lm_recurrent)

                            if l < self._num_layers - 1:
                                lm_topdown_in_dim = top_down_dim
                                kernel_lm_topdown = self.add_variable(
                                    'kernel_lm_topdown_%d' % l,
                                    shape=[lm_topdown_in_dim, bottom_up_dim],
                                    initializer=self._recurrent_initializer
                                )
                                if self._weight_normalization:
                                    kernel_lm_topdown_g = tf.Variable(tf.ones([1, bottom_up_dim]), name='kernel_lm_topdown_g_%d' % l)
                                    kernel_lm_topdown_b = tf.Variable(tf.zeros([1, bottom_up_dim]), name='kernel_lm_topdown_b_%d' % l)
                                    kernel_lm_topdown = tf.nn.l2_normalize(kernel_lm_topdown, axis=-1) * kernel_lm_topdown_g + kernel_lm_topdown_b
                                self._regularize(kernel_lm_topdown, self._topdown_regularizer)
                                self._kernel_lm_topdown.append(kernel_lm_topdown)

                            bias_lm = self.add_variable(
                                'bias_lm_%d' % l,
                                shape=[1, bottom_up_dim],
                                initializer=self._bias_initializer
                            )
                            self._regularize(bias_lm, self._bias_regularizer)
                            self._bias_lm.append(bias_lm)

                        if not self._layer_normalization:
                            bias = self.add_variable(
                                'bias_%d' % l,
                                shape=[1, output_dim],
                                initializer=self._bias_initializer
                            )
                            self._regularize(bias, self._bias_regularizer)
                            self._bias.append(bias)

                        if not self._oracle_boundary:
                            if self._implementation == 1 and not self._layer_normalization:
                                bias_boundary = bias[:, -1:]
                                self._bias_boundary.append(bias_boundary)

                            if self._implementation == 2:
                                kernel_boundary = self.add_variable(
                                    'kernel_boundary_%d' % l,
                                    shape=[self._num_units[l] + self._refeed_boundary, 1],
                                    initializer=self._boundary_initializer
                                )
                                if self._weight_normalization:
                                    kernel_boundary_g = tf.Variable(tf.ones([1, output_dim]), name='kernel_boundary_g_%d' % l)
                                    kernel_boundary_b = tf.Variable(tf.zeros([1, output_dim]), name='kernel_boundary_b_%d' % l)
                                    kernel_boundary = tf.nn.l2_normalize(kernel_boundary, axis=-1) * kernel_boundary_g + kernel_boundary_b
                                self._regularize(kernel_boundary, self._boundary_regularizer)
                                self._kernel_boundary.append(kernel_boundary)

                                bias_boundary = self.add_variable(
                                    'bias_boundary_%d' % l,
                                    shape=[1, 1],
                                    initializer=self._bias_initializer
                                )
                                self._regularize(bias_boundary, self._bias_regularizer)
                                self._bias_boundary.append(bias_boundary)

                    if not self._oracle_boundary:
                        if self._boundary_slope_annealing_rate and self.global_step is not None:
                            rate = self._boundary_slope_annealing_rate
                            if self._slope_annealing_max is None:
                                self.boundary_slope_coef = 1 + rate * tf.cast(self.global_step, dtype=tf.float32)
                            else:
                                self.boundary_slope_coef = tf.minimum(self._slope_annealing_max, 1 + rate * tf.cast(self.global_step, dtype=tf.float32))
                        else:
                            self.boundary_slope_coef = None
                    else:
                        self.boundary_slope_coef = None

                    if self._state_slope_annealing_rate and self.global_step is not None:
                        rate = self._state_slope_annealing_rate
                        if self._slope_annealing_max is None:
                            self.state_slope_coef = 1 + rate * tf.cast(self.global_step, dtype=tf.float32)
                        else:
                            self.state_slope_coef = tf.minimum(self._slope_annealing_max, 1 + rate * tf.cast(self.global_step, dtype=tf.float32))
                    else:
                        self.state_slope_coef = None

        self.built = True

    def call(self, inputs, state):
        with self._session.as_default():
            with self._session.graph.as_default():
                with tf.device(self._device):
                    new_output = []
                    new_state = []

                    if self._oracle_boundary:
                        n_boundary_dims = self._num_layers - 1
                        h_below = inputs[:, :-n_boundary_dims]
                    else:
                        h_below = inputs

                    z_below = None

                    if self._power:
                        power = self._power
                    else:
                        power = 1

                    for l, layer in enumerate(state):
                        # z_behind: Previous boundary probability at current layer (implicitly 1 if final layer)
                        if l < self._num_layers - 1:
                            z_behind = layer[2]
                        else:
                            z_behind = None

                        # Compute probability of update, copy, and flush operations operations.
                        if z_behind is None:
                            z_behind_cur = 0
                        else:
                            z_behind_cur = z_behind

                        if z_below is None:
                            z_below_cur = 1
                        else:
                            z_below_cur = z_below

                        update_prob = (1 - z_behind_cur) * z_below_cur
                        copy_prob = (1 - z_behind_cur) * (1 - z_below_cur)
                        flush_prob = z_behind_cur
                        if power != 1:
                            # Attention only sums to 1 if probs are 0 or 1, discourages middling attention weights
                            update_prob = update_prob ** power
                            copy_prob = copy_prob ** power
                            flush_prob = flush_prob ** power

                        # Alias useful variables
                        if l < self._num_layers - 1:
                            # Use inner activation if non-final layer
                            activation = self._inner_activation
                        else:
                            # Use outer activation if final layer
                            activation = self._activation
                        units = self._num_units[l]

                        # EXTRACT DEPENDENCIES (c_behind, h_behind, h_above, h_below, z_behind, z_below):

                        # c_behind: Previous cell state at current layer
                        c_behind = layer[0]

                        # h_behind: Previous hidden state at current layer
                        h_behind = layer[1]
                        h_behind = self._recurrent_dropout(h_behind)

                        # h_above: Hidden state of layer above at previous timestep (implicitly 0 if final layer)
                        if l < self._num_layers - 1:
                            h_above = state[l + 1][1]
                            h_above = self._topdown_dropout(h_above)
                        else:
                            h_above = None

                        # h_below: Bottom-up input
                        if self._state_discretizer and l > 0:
                            h_below = 2 * (h_below - 0.5)
                        if self._lm:
                            lm_logits = tf.matmul(
                                tf.concat([c_behind, h_behind], axis=-1),
                                self._kernel_lm_recurrent[l]
                            )
                            if z_behind is not None:
                                lm_logits *= (1 - z_behind ** power)
                                lm_logits += tf.matmul(h_above, self._kernel_lm_topdown[l]) * (z_behind ** power)

                            lm_logits += self._bias_lm[l]

                            if self._one_hot_inputs and l == 0:
                                lm = tf.nn.softmax(lm_logits)
                                if self._return_lm_loss:
                                    lm_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                                        labels=tf.stop_gradient(h_below),
                                        logits=lm_logits
                                    )[..., None]
                            elif self._state_discretizer is None:
                                lm = self._inner_activation(lm_logits)
                                if self._return_lm_loss:
                                    lm_loss = tf.reduce_mean(
                                        (tf.stop_gradient(tf.atanh(h_below*(1-1e-9))) - lm_logits) ** 2,
                                        axis=-1,
                                        keep_dims=True
                                    )
                            else:
                                lm = tf.sigmoid(lm_logits)
                                if self._return_lm_loss:
                                    lm_loss = tf.reduce_mean(
                                        (tf.stop_gradient(h_below) - lm) ** 2,
                                        axis=-1,
                                        keep_dims=True
                                    )
                        if self._implementation == 1 and self._refeed_boundary:
                            h_below = tf.concat([z_behind, h_below], axis=1)
                        if self._temporal_dropout[l]:
                            def train_func():
                                # data = (h_below + lm) / 2.
                                data = h_below
                                noise = tf.random_uniform([tf.shape(h_below)[0]]) > self._temporal_dropout[l]
                                if self._temporal_dropout_plug_lm:
                                    # if self._one_hot_inputs and l == 0:
                                    #     alt = tf.one_hot(tf.argmax(lm, axis=-1), lm.shape[-1])
                                    # else:
                                    #     alt = lm
                                    alt = lm
                                else:
                                    alt = tf.zeros_like(h_below)
                                out = tf.where(noise, data, alt)
                                return out

                            def eval_func():
                                # return (h_below + lm) / 2.
                                return h_below

                            h_below = tf.cond(self._training, train_func, eval_func)
                        elif self._lm:
                            pass
                            # h_below = (h_below + lm) / 2.

                        # h_below = self._temporal_dropout[l](h_below)
                        h_below = self._bottomup_dropout(h_below)

                        s_bottomup = tf.matmul(h_below, self._kernel_bottomup[l])
                        if l > 0:
                            s_bottomup = s_bottomup * (z_below ** power)

                        # Recurrent features
                        if self._state_discretizer and l < self._num_layers - 1:
                            s_recurrent_in = 2 * (h_behind - 0.5)
                        else:
                            s_recurrent_in = h_behind
                        s_recurrent = tf.matmul(s_recurrent_in, self._kernel_recurrent[l])

                        # Sum bottom-up and recurrent features
                        s = s_bottomup + s_recurrent

                        # Top-down features (if non-final layer)
                        if l < self._num_layers - 1:
                            # Compute top-down features
                            s_topdown = tf.matmul(h_above, self._kernel_topdown[l]) * (z_behind ** power)
                            # Add in top-down features
                            s = s + s_topdown

                        if self._implementation == 1:
                            # In implementation 1, boundary has its own slice of the hidden state preactivations
                            z = s[:, units * 4:]
                            s = s[:, :units * 4]

                        if self._layer_normalization:
                            s = self.norm(s, 's_ln_%d' % l)
                        else:
                            if self._implementation == 1:
                                z += self._bias[l][:, units * 4:]
                            s = s + self._bias[l][:, :units * 4]

                        # Forget gate
                        f = s[:, :units]
                        f = self._recurrent_activation(f + self._forget_bias)

                        # Input gate
                        i = s[:, units:units * 2]
                        i = self._recurrent_activation(i)

                        # Output gate
                        o = s[:, units * 2:units * 3]
                        o = self._recurrent_activation(o)

                        # Cell proposal
                        g = s[:, units * 3:units * 4]
                        g = activation(g)

                        # Cell state update (forget-gated previous cell plus input-gated cell proposal)
                        c_update = f * c_behind + i * g
                        if self._layer_normalization:
                            c_update = self.norm(c_update, 'c_%d' % l)

                        # Compute cell state copy operation
                        c_copy = c_behind

                        # Compute cell state flush operation
                        c_flush = i * g

                        # Merge cell operations. If boundaries are hard, selects between update, copy, and flush.
                        # If boundaries are soft, sums update copy and flush proportionally to their probs.
                        c = update_prob * c_update + copy_prob * c_copy + flush_prob * c_flush

                        # Compute the gated output
                        if l < self._num_layers - 1 and self._state_discretizer is not None:
                            activation = tf.sigmoid
                            # activation = tf.keras.backend.hard_sigmoid
                        if self._state_slope_annealing_rate and self.global_step is not None:
                            h = o * activation(c * self.state_slope_coef)
                        else:
                            h = o * activation(c)
                        if l < self._num_layers - 1 and self._state_discretizer and self.global_step is not None:
                            h = self._state_discretizer(h)

                        # Mix gated output with the previous hidden state proportionally to the copy probability
                        h = copy_prob * h_behind + (1 - copy_prob) * h

                        # Compute the current boundary probability
                        if l < self._num_layers - 1:
                            if self._oracle_boundary:
                                inputs_last_dim = inputs.shape[-1]
                                z_prob = inputs[:, inputs_last_dim + l - 1:inputs_last_dim + l]
                                z = z_prob
                            else:
                                if self._implementation == 2:
                                    # In implementation 2, boundary is computed by linear transform of h
                                    z_in = h
                                    if self._refeed_boundary:
                                        z_in = tf.concat([z_behind, z_in], axis=1)
                                    z_in = self._boundary_dropout(z_in)
                                    # In implementation 2, boundary is a function of the hidden state
                                    z = tf.matmul(z_in, self._kernel_boundary[l])
                                    z = z + self._bias_boundary[l]

                                if self._boundary_slope_annealing_rate and self.global_step is not None:
                                    z *= self.boundary_slope_coef

                                z_prob = self._boundary_activation(z)

                                # if l > 0:
                                #     z_prob = z_prob * z_below

                                if self._boundary_discretizer:
                                    z = self._boundary_discretizer(z_prob)
                                else:
                                    z = z_prob
                        else:
                            z_prob = None
                            z = None

                        if l < self._num_layers - 1:
                            if self._return_lm_loss:
                                output_l = (h, z_prob, z, lm_loss)
                            else:
                                output_l = (h, z_prob, z)
                            new_state_l = (c, h, z)
                        else:
                            if self._return_lm_loss:
                                output_l = (h, lm_loss)
                            else:
                                output_l = (h,)
                            new_state_l = (c, h,)

                        # Append layer to output and state
                        new_output.append(output_l)
                        new_state.append(new_state_l)

                        h_below = h
                        z_below = z

                    # Create output and state tuples
                    new_output = tuple(new_output)
                    new_state = tuple(new_state)

                    return new_output, new_state


class HMLSTMSegmenter(object):
    def __init__(
            self,
            num_units,
            num_layers,
            training=False,
            one_hot_inputs=False,
            forget_bias=1.0,
            oracle_boundary=False,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            boundary_activation='sigmoid',
            boundary_discretizer=None,
            bottomup_initializer='glorot_normal_initializer',
            recurrent_initializer='orthogonal_initializer',
            topdown_initializer='glorot_normal_initializer',
            boundary_initializer='orthogonal_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            topdown_regularizer=None,
            boundary_regularizer=None,
            bias_regularizer=None,
            bottomup_dropout=None,
            temporal_dropout=None,
            temporal_dropout_plug_lm=False,
            return_lm_loss=False,
            recurrent_dropout=None,
            topdown_dropout=None,
            boundary_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            refeed_boundary=False,
            power=None,
            boundary_slope_annealing_rate=None,
            state_slope_annealing_rate=None,
            slope_annealing_max=None,
            state_discretizer=None,
            sample_at_train=True,
            sample_at_eval=False,
            global_step=None,
            implementation=1,
            reuse=None,
            name=None,
            dtype=None,
            session=None,
            device=None
    ):
        self.session = get_session(session)
        self.device = device

        with self.session.as_default():
            with self.session.graph.as_default():
                with tf.device(self.device):
                    if not isinstance(num_units, list):
                        self.num_units = [num_units] * num_layers
                    else:
                        self.num_units = num_units

                    assert len(
                        self.num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                    self.num_layers = num_layers
                    self.training = training
                    self.one_hot_inputs = one_hot_inputs
                    self.forget_bias = forget_bias
                    self.oracle_boundary = oracle_boundary

                    self.activation = activation
                    self.inner_activation = inner_activation
                    self.recurrent_activation = recurrent_activation
                    self.boundary_activation = boundary_activation
                    self.boundary_discretizer = boundary_discretizer

                    self.bottomup_initializer = bottomup_initializer
                    self.recurrent_initializer = recurrent_initializer
                    self.topdown_initializer = topdown_initializer
                    self.boundary_initializer = boundary_initializer
                    self.bias_initializer = bias_initializer

                    self.temporal_dropout = temporal_dropout
                    self.return_lm_loss = return_lm_loss
                    self.temporal_dropout_plug_lm = temporal_dropout_plug_lm
                    self.bottomup_dropout = bottomup_dropout
                    self.recurrent_dropout = recurrent_dropout
                    self.topdown_dropout = topdown_dropout
                    self.boundary_dropout = boundary_dropout

                    self.bottomup_regularizer = bottomup_regularizer
                    self.recurrent_regularizer = recurrent_regularizer
                    self.topdown_regularizer = topdown_regularizer
                    self.boundary_regularizer = boundary_regularizer
                    self.bias_regularizer = bias_regularizer

                    self.weight_normalization = weight_normalization
                    self.layer_normalization = layer_normalization
                    self.refeed_boundary = refeed_boundary
                    self.power = power
                    self.boundary_slope_annealing_rate = boundary_slope_annealing_rate
                    self.state_slope_annealing_rate = state_slope_annealing_rate
                    self.slope_annealing_max = slope_annealing_max
                    self.state_discretizer = state_discretizer
                    self.sample_at_train = sample_at_train
                    self.sample_at_eval = sample_at_eval
                    self.global_step = global_step
                    self.implementation = implementation

                    self.reuse = reuse
                    self.name = name
                    self.dtype = dtype
                    self.device = device
                    self.boundary_slope_coef = None
                    self.state_slope_coef = None

                    self.built = False

    def get_regularizer_losses(self):
        if self.built:
            return self.cell.get_regularizer_losses()
        raise ValueError(
            'Attempted to get regularizer losses from an HMLSTMSegmenter that has not been called on data.')

    def build(self, inputs=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                with tf.device(self.device):
                    self.cell = HMLSTMCell(
                        self.num_units,
                        self.num_layers,
                        training=self.training,
                        one_hot_inputs=self.one_hot_inputs,
                        forget_bias=self.forget_bias,
                        oracle_boundary=self.oracle_boundary,
                        activation=self.activation,
                        inner_activation=self.inner_activation,
                        recurrent_activation=self.recurrent_activation,
                        boundary_activation=self.boundary_activation,
                        boundary_discretizer=self.boundary_discretizer,
                        bottomup_initializer=self.bottomup_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        topdown_initializer=self.topdown_initializer,
                        bias_initializer=self.bias_initializer,
                        bottomup_regularizer=self.bottomup_regularizer,
                        recurrent_regularizer=self.recurrent_regularizer,
                        topdown_regularizer=self.topdown_regularizer,
                        boundary_regularizer=self.boundary_regularizer,
                        temporal_dropout=self.temporal_dropout,
                        return_lm_loss=self.return_lm_loss,
                        temporal_dropout_plug_lm=self.temporal_dropout_plug_lm,
                        bottomup_dropout=self.bottomup_dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        topdown_dropout=self.topdown_dropout,
                        boundary_dropout=self.boundary_dropout,
                        bias_regularizer=self.bias_regularizer,
                        weight_normalization=self.weight_normalization,
                        layer_normalization=self.layer_normalization,
                        refeed_boundary=self.refeed_boundary,
                        power=self.power,
                        boundary_slope_annealing_rate=self.boundary_slope_annealing_rate,
                        state_slope_annealing_rate=self.state_slope_annealing_rate,
                        slope_annealing_max=self.slope_annealing_max,
                        state_discretizer=self.state_discretizer,
                        sample_at_train=self.sample_at_train,
                        sample_at_eval=self.sample_at_eval,
                        global_step=self.global_step,
                        implementation=self.implementation,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        session=self.session,
                        device=self.device
                    )

                    self.cell.build(inputs.shape[1:])

        self.built = True

    def __call__(self, inputs, mask=None, boundaries=None):
        assert self.oracle_boundary != (boundaries is None), 'A boundaries arg must be provided always and only when oracle_boundary is true'

        with self.session.as_default():
            with self.session.graph.as_default():
                with tf.device(self.device):
                    if mask is not None:
                        sequence_length = tf.reduce_sum(mask, axis=1)
                    else:
                        sequence_length = None

                    if self.oracle_boundary:
                        assert boundaries.shape[-1] == (self.num_layers - 1)
                        inputs = tf.concat([inputs, boundaries], axis=-1)

                    if not self.built:
                        self.build(inputs)

                    output, state = tf.nn.dynamic_rnn(
                        self.cell,
                        inputs,
                        sequence_length=sequence_length,
                        swap_memory=True,
                        dtype=tf.float32
                    )

                    self.boundary_slope_coef = self.cell.boundary_slope_coef
                    self.state_slope_coef = self.cell.state_slope_coef

                    out = HMLSTMOutput(output)

                    return out


class HMLSTMOutput(object):
    def __init__(self, output, session=None):
        self.session = get_session(session)

        self.num_layers = len(output)
        self.l = [HMLSTMOutputLevel(level, session=self.session) for level in output]

    def state(self, level=None, discrete=False, method='round', mask=None):
        if level is None:
            out = tuple([l.state(discrete=discrete, discretization_method=method, mask=mask) for l in self.l])
        else:
            out = self.l[level].state(discrete=discrete, discretization_method=method, mask=mask)

        return out

    def boundary(self, level=None, discrete=False, method='round', as_logits=False, mask=None):
        if level is None:
            out = tuple(
                [l.boundary(discrete=discrete, discretization_method=method, as_logits=as_logits, mask=mask) for l in self.l[:-1]])
        else:
            out = self.l[level].boundary(discrete=discrete, discretization_method=method, as_logits=as_logits, mask=mask)

        return out

    def boundary_probs(self, level=None, as_logits=False, mask=None):
        if level is None:
            out = tuple([l.boundary_probs(as_logits=as_logits, mask=mask) for l in self.l[:-1]])
        else:
            out = self.l[level].boundary_probs(as_logits=as_logits, mask=mask)

        return out

    def output(self, all_layers=False, return_sequences=False):
        if all_layers:
            out = self.state(level=None, discrete=False)
            out = tf.concat(out, axis=-1)
        else:
            out = self.state(level=-1, discrete=False)

        if not return_sequences:
            out = out[:, -1, :]

        return out

    def lm_loss(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                loss = [l.lm_loss(mask=mask) for l in self.l]

                return loss


class HMLSTMOutputLevel(object):
    def __init__(self, output, session=None):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                self.h = output[0]
                if len(output) > 2:
                    self.z_prob = output[1]
                    self.z = output[2]
                    if len(output) > 3:
                        self.lm_losses = output[3]
                    else:
                        self.lm_losses = None
                else:
                    self.z_prob = None
                    self.z = None
                    if len(output) > 1:
                        self.lm_losses = output[1]
                    else:
                        self.lm_losses = None

    def state(self, discrete=False, discretization_method='round', mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.h
                if discrete:
                    if discretization_method == 'round':
                        out = tf.cast(tf.round(self.h), dtype=tf.int32)
                    else:
                        raise ValueError('Discretization method "%s" not currently supported' % discretization_method)

                if mask is not None:
                    while len(mask.shape) < len(out.shape):
                        mask = mask[..., None]
                    out = out * mask

                return out

    def boundary(self, discrete=False, discretization_method='round', as_logits=False, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                if self.z is None:
                    out = tf.zeros(tf.shape(self.h)[:2])
                else:
                    out = self.z[..., 0]
                    if not discrete and as_logits:
                        out = sigmoid_to_logits(out, session=self.session)

                if discrete:
                    if discretization_method == 'round':
                        out = tf.round(out)
                    else:
                        raise ValueError('Discretization method "%s" not currently supported' % discretization_method)

                    out = tf.cast(out, dtype=tf.int32)

                if mask is not None:
                    while len(mask.shape) < len(out.shape):
                        mask = mask[..., None]
                    out = out * mask

                return out

    def boundary_probs(self, as_logits=False, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                if self.z_prob is None:
                    out = tf.zeros(tf.shape(self.h)[:2])
                else:
                    out = self.z_prob[..., 0]
                    if as_logits:
                        out = sigmoid_to_logits(out, session=self.session)

                if mask is not None:
                    while len(mask.shape) < len(out.shape):
                        mask = mask[..., None]
                    out = out * mask

                return out

    def lm_loss(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                losses = self.lm_losses
                if losses is not None:
                    if mask is None:
                        loss = tf.reduce_mean(losses)
                    else:
                        losses *= mask[..., None]
                        loss = tf.reduce_sum(losses) / (tf.reduce_sum(mask) + 1e-8)

                else:
                    loss = tf.zeros([])

                # loss = tf.zeros([])

                return loss


class MultiLSTMCell(LayerRNNCell):
    def __init__(
            self,
            num_units,
            num_layers,
            training=True,
            forget_bias=1.0,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs=False,
            reuse=None,
            name=None,
            dtype=None,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                super(MultiLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                if not isinstance(num_units, list):
                    self._num_units = [num_units] * num_layers
                else:
                    self._num_units = num_units

                assert len(self._num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                self._num_layers = num_layers
                self._forget_bias = forget_bias

                self._training=training

                self._activation = get_activation(activation, session=self.session, training=self._training)
                self._inner_activation = get_activation(inner_activation, session=self.session, training=self._training)
                self._recurrent_activation = get_activation(recurrent_activation, session=self.session, training=self._training)

                self._kernel_initializer = get_initializer(kernel_initializer, session=self.session)
                self._bias_initializer = get_initializer(bias_initializer, session=self.session)

                self._refeed_outputs = refeed_outputs

    def _regularize(self, var, regularizer):
        if regularizer is not None:
            with self.session.as_default():
                with self.session.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)

    @property
    def state_size(self):
        out = []
        for l in range(self._num_layers):
            size = (self._num_units[l], self._num_units[l])
            out.append(size)

        out = tuple(out)

        return out

    @property
    def output_size(self):
        out = self._num_units[-1]

        return out

    def build(self, inputs_shape):
        with self.session.as_default():
            with self.session.graph.as_default():
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                     % inputs_shape)

                self._kernel = []
                self._bias = []

                for l in range(self._num_layers):
                    if l == 0:
                        bottom_up_dim = inputs_shape[1].value
                    else:
                        bottom_up_dim = self._num_units[l-1]

                    recurrent_dim = self._num_units[l]
                    output_dim = 4 * self._num_units[l]
                    if self._refeed_outputs and l == 0:
                        refeed_dim = self._num_units[-1]
                    else:
                        refeed_dim = 0

                    kernel = self.add_variable(
                        'kernel_%d' %l,
                        shape=[bottom_up_dim + recurrent_dim + refeed_dim, output_dim],
                        initializer=self._kernel_initializer
                    )
                    self._kernel.append(kernel)

                    bias = self.add_variable(
                        'bias_%d' %l,
                        shape=[1, output_dim],
                        initializer=self._bias_initializer
                    )
                    self._bias.append(bias)

        self.built = True

    def call(self, inputs, state):
        with self.session.as_default():
            new_state = []

            h_below = inputs
            for l, layer in enumerate(state):
                c_behind, h_behind = layer

                # Gather inputs
                layer_inputs = [h_below, h_behind]

                if self._refeed_outputs and l == 0:
                    prev_outputs = state[-1][1]
                    layer_inputs.append(prev_outputs)

                layer_inputs = tf.concat(layer_inputs, axis=1)

                # Compute gate pre-activations
                s = tf.matmul(
                    layer_inputs,
                    self._kernel[l]
                )

                # Add bias
                s = s + self._bias[l]

                # Alias useful variables
                if l < self._num_layers - 1:
                    # Use inner activation if non-final layer
                    activation = self._inner_activation
                else:
                    # Use outer activation if final layer
                    activation = self._activation
                units = self._num_units[l]

                # Forget gate
                f = self._recurrent_activation(s[:, :units] + self._forget_bias)
                # Input gate
                i = self._recurrent_activation(s[:, units:units * 2])
                # Output gate
                o = self._recurrent_activation(s[:, units * 2:units * 3])
                # Cell proposal
                g = activation(s[:, units * 3:units * 4])

                # Compute new cell state
                c = f * c_behind + i * g

                # Compute the gated output
                h = o * activation(c)

                new_state.append((c, h))

                h_below = h

            new_state = tuple(new_state)
            new_output = new_state[-1][1]

            return new_output, new_state


class DenseLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            activation=None,
            batch_normalization_decay=0.9,
            normalize_weights=False,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.batch_normalization_decay = batch_normalization_decay
        self.normalize_weights = normalize_weights

        self.dense_layer = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            if self.units is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.units

            with self.session.as_default():
                with self.session.graph.as_default():
                    self.dense_layer = tf.keras.layers.Dense(
                        out_dim,
                        input_shape=[inputs.shape[1]],
                        use_bias=self.use_bias,
                        kernel_initializer='glorot_normal',
                    )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.dense_layer(inputs)

                if self.normalize_weights:
                    self.w = self.dense_layer.kernel
                    self.g = tf.Variable(tf.ones(self.w.shape[1]), dtype=tf.float32)
                    self.v = tf.norm(self.w, axis=0)
                    self.dense_layer.kernel = self.v

                if self.batch_normalization_decay:
                    # H = tf.layers.batch_normalization(H, training=self.training)
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )
                if self.activation is not None:
                    H = self.activation(H)

                return H


class DenseResidualLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            layers_inner=3,
            activation_inner=None,
            activation=None,
            batch_normalization_decay=0.9,
            project_inputs=False,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.layers_inner = layers_inner
        self.activation_inner = get_activation(activation_inner, session=self.session, training=self.training)
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs

        self.dense_layers = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.units is None:
                        out_dim = inputs.shape[-1]
                    else:
                        out_dim = self.units

                    self.dense_layers = []

                    for i in range(self.layers_inner):
                        if i == 0:
                            in_dim = inputs.shape[1]
                        else:
                            in_dim = out_dim
                        l = tf.keras.layers.Dense(
                            out_dim,
                            input_shape=[in_dim],
                            use_bias=self.use_bias
                        )
                        self.dense_layers.append(l)

                    if self.project_inputs:
                        self.projection = tf.keras.layers.Dense(
                            out_dim,
                            input_shape=[inputs.shape[1]]
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                F = inputs
                for i in range(self.layers_inner - 1):
                    F = self.dense_layers[i](F)
                    if self.batch_normalization_decay:
                        # F = tf.layers.batch_normalization(F, training=self.training)
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.dense_layers[-1](F)
                if self.batch_normalization_decay:
                    # F = tf.layers.batch_normalization(F, training=self.training)
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                if self.project_inputs:
                    x = self.projection(inputs)
                else:
                    x = inputs

                H = F + x

                if self.activation is not None:
                    H = self.activation(H)

                return H


class Conv1DLayer(object):

    def __init__(
            self,
            kernel_size,
            training=True,
            n_filters=None,
            stride=1,
            padding='valid',
            use_bias=True,
            activation=None,
            dropout=None,
            batch_normalization_decay=0.9,
            session=None
    ):
        self.session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                self.training = training
                self.n_filters = n_filters
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.use_bias = use_bias
                self.activation = get_activation(activation, session=self.session, training=self.training)
                self.dropout = get_dropout(dropout, session=self.session, noise_shape=None, training=self.training)
                self.batch_normalization_decay = batch_normalization_decay

                self.conv_1d_layer = None

                self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.n_filters is None:
                        out_dim = inputs.shape[-1]
                    else:
                        out_dim = self.n_filters

                    self.conv_1d_layer = tf.keras.layers.Conv1D(
                        out_dim,
                        self.kernel_size,
                        padding=self.padding,
                        strides=self.stride,
                        use_bias=self.use_bias
                    )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                H = inputs

                H = self.conv_1d_layer(H)

                if self.batch_normalization_decay:
                    # H = tf.layers.batch_normalization(H, training=self.training)
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                if self.activation is not None:
                    H = self.activation(H)

                return H


class Conv1DResidualLayer(object):

    def __init__(
            self,
            kernel_size,
            training=True,
            n_filters=None,
            stride=1,
            padding='valid',
            use_bias=True,
            layers_inner=3,
            activation=None,
            activation_inner=None,
            batch_normalization_decay=0.9,
            project_inputs=False,
            n_timesteps=None,
            n_input_features=None,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.layers_inner = layers_inner
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.activation_inner = get_activation(activation_inner, session=self.session, training=self.training)
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs
        self.n_timesteps = n_timesteps
        self.n_input_features = n_input_features

        self.conv_1d_layers = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            if self.n_filters is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.n_filters

            self.built = True

            self.conv_1d_layers = []

            with self.session.as_default():
                with self.session.graph.as_default():

                    conv_output_shapes = [[int(inputs.shape[1]), int(inputs.shape[2])]]

                    for i in range(self.layers_inner):
                        if isinstance(self.stride, list):
                            cur_strides = self.stride[i]
                        else:
                            cur_strides = self.stride

                        l = tf.keras.layers.Conv1D(
                            out_dim,
                            self.kernel_size,
                            padding=self.padding,
                            strides=cur_strides,
                            use_bias=self.use_bias
                        )

                        if self.padding in ['causal', 'same'] and self.stride == 1:
                            output_shape = conv_output_shapes[-1]
                        else:
                            output_shape = [
                                conv_output_length(
                                    x,
                                    self.kernel_size,
                                    self.padding,
                                    self.stride
                                ) for x in conv_output_shapes[-1]
                            ]

                        conv_output_shapes.append(output_shape)

                        self.conv_1d_layers.append(l)

                    self.conv_output_shapes = conv_output_shapes

                    if self.project_inputs:
                        self.projection = tf.keras.layers.Dense(
                            self.conv_output_shapes[-1][0] * out_dim,
                            input_shape=[self.conv_output_shapes[0][0] * self.conv_output_shapes[0][1]]
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                F = inputs

                for i in range(self.layers_inner - 1):
                    F = self.conv_1d_layers[i](F)

                    if self.batch_normalization_decay:
                        # F = tf.layers.batch_normalization(F, training=self.training)
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.conv_1d_layers[-1](F)

                if self.batch_normalization_decay:
                    # F = tf.layers.batch_normalization(F, training=self.training)
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                if self.project_inputs:
                    x = tf.layers.Flatten()(inputs)
                    x = self.projection(x)
                    x = tf.reshape(x, tf.shape(F))
                else:
                    x = inputs

                H = F + x

                if self.activation is not None:
                    H = self.activation(H)

                return H


class RNNLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            activation=None,
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs=False,
            return_sequences=True,
            batch_normalization_decay=None,
            name=None,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.refeed_outputs = refeed_outputs
        assert not (return_sequences and batch_normalization_decay), 'batch_normalization_decay can only be used when return_sequences=False'
        self.return_sequences = return_sequences
        self.batch_normalization_decay = batch_normalization_decay
        self.name = name

        self.rnn_layer = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    RNN = tf.keras.layers.LSTM

                    if self.units:
                        output_dim = self.units
                    else:
                        output_dim = inputs.shape[-1]

                    self.rnn_layer = RNN(
                        output_dim,
                        return_sequences=self.return_sequences,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation
                    )

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.rnn_layer(inputs, mask=mask)
                if self.batch_normalization_decay:
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                return H


class MultiRNNLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            layers=1,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs=False,
            return_sequences=True,
            name=None,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.layers = layers
        self.activation = activation
        self.inner_activation = inner_activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.refeed_outputs = refeed_outputs
        self.return_sequences = return_sequences
        self.name = name

        self.rnn_layer = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    # RNN = getattr(tf.keras.layers, self.rnn_type)

                    if self.units is None:
                        units = [inputs.shape[-1]] * self.layers
                    else:
                        units = self.units

                    # self.rnn_layer = RNN(
                    #     out_dim,
                    #     return_sequences=self.return_sequences,
                    #     activation=self.activation,
                    #     recurrent_activation=self.recurrent_activation
                    # )
                    # self.rnn_layer = tf.contrib.rnn.BasicLSTMCell(
                    #     out_dim,
                    #     activation=self.activation,
                    #     name=self.name
                    # )

                    self.rnn_layer = MultiLSTMCell(
                        units,
                        self.layers,
                        training=self.training,
                        activation=self.activation,
                        inner_activation=self.inner_activation,
                        recurrent_activation=self.recurrent_activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        refeed_outputs=self.refeed_outputs,
                        name=self.name,
                        session=self.session
                    )

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                # H = self.rnn_layer(inputs, mask=mask)
                if mask is None:
                    sequence_length = None
                else:
                    sequence_length = tf.reduce_sum(mask, axis=1)

                H, _ = tf.nn.dynamic_rnn(
                    self.rnn_layer,
                    inputs,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

                if not self.return_sequences:
                    H = H[:,-1]

                return H


def rnn_encoder(
        n_feats_in,
        units_encoder,
        training=True,
        pre_cnn=False,
        cnn_kernel_size=5,
        inner_activation='tanh',
        activation='tanh',
        recurrent_activation='sigmoid',
        batch_normalization_decay=None,
        session=None
):
    session = get_session(session)
    n_layers = len(units_encoder)
    lambdas = []
    kwargs = {'mask': None}

    with session.as_default():
        with session.graph.as_default():
            if pre_cnn:
                cnn_layer = Conv1DLayer(
                    cnn_kernel_size,
                    training=training,
                    n_filters=n_feats_in,
                    padding='same',
                    activation=tf.nn.elu,
                    batch_normalization_decay=batch_normalization_decay,
                    session=session
                )
                lambdas.append(make_lambda(cnn_layer, session=None))


            for l in range(n_layers):
                if l < n_layers - 1:
                    activation = inner_activation
                    batch_normalization_decay_cur = None
                else:
                    activation = activation
                    batch_normalization_decay_cur = batch_normalization_decay

                rnn_layer = RNNLayer(
                    training=training,
                    units=units_encoder[l],
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    return_sequences=False,
                    batch_normalization_decay=batch_normalization_decay_cur,
                    name='RNNEncoder%s' % l,
                    session=session
                )

                lambdas.append(make_lambda(rnn_layer, session=None, use_kwargs=True))

            out = compose_lambdas(lambdas, **kwargs)

            return out


def cnn_encoder(
        n_feats_in,
        kernel_size,
        units_encoder,
        training=True,
        inner_activation='tanh',
        activation='tanh',
        resnet_n_layers_inner=None,
        batch_normalization_decay=None,
        session=None
):
    session = get_session(session)
    n_layers = len(units_encoder)
    lambdas = []
    kwargs = {'mask': None} # The mask param isn't used. It's just included to permit a unified encoder interface.

    with session.as_default():
        with session.graph.as_default():
            cnn_layer = Conv1DLayer(
                kernel_size,
                training=training,
                n_filters=n_feats_in,
                padding='same',
                activation=tf.nn.elu,
                batch_normalization_decay=batch_normalization_decay,
                session=session
            )

            def apply_cnn_layer(x, **kwargs):
                return cnn_layer(x)
            lambdas.append(apply_cnn_layer)

            for i in range(n_layers - 1):
                if i > 0 and resnet_n_layers_inner:
                    cnn_layer = Conv1DResidualLayer(
                        kernel_size,
                        training=training,
                        n_filters=units_encoder[i],
                        padding='causal',
                        layers_inner=resnet_n_layers_inner,
                        activation=inner_activation,
                        activation_inner=inner_activation,
                        batch_normalization_decay=batch_normalization_decay,
                        session=session
                    )
                else:
                    cnn_layer = Conv1DLayer(
                        kernel_size,
                        training=training,
                        n_filters=units_encoder[i],
                        padding='causal',
                        activation=inner_activation,
                        batch_normalization_decay=batch_normalization_decay,
                        session=session
                    )

                def apply_cnn_layer(x, **kwargs):
                    return cnn_layer(x)
                lambdas.append(apply_cnn_layer)

            flattener = tf.layers.Flatten()
            lambdas.append(make_lambda(flattener, session=None))

            fully_connected_layer = DenseLayer(
                training=training,
                units=units_encoder[-1],
                activation=activation,
                batch_normalization_decay=batch_normalization_decay,
                session=session
            )
            lambdas.append(make_lambda(fully_connected_layer, session=None))

            out = compose_lambdas(lambdas, **kwargs)

            return out


def dense_encoder(
        n_timesteps,
        units_encoder,
        training=True,
        inner_activation='tanh',
        activation='tanh',
        resnet_n_layers_inner=None,
        batch_normalization_decay=None,
        session=None
):
    session = get_session(session)
    n_layers = len(units_encoder)
    lambdas = []
    kwargs = {'mask': None}  # The mask param isn't used. It's just included to permit a unified encoder interface.

    with session.as_default():
        with session.graph.as_default():
            flattener = tf.layers.Flatten()
            lambdas.append(make_lambda(flattener, session=None))

            for i in range(n_layers - 1):
                if i > 0 and resnet_n_layers_inner:
                    if units_encoder[i] != units_encoder[i - 1]:
                        project_inputs = True
                    else:
                        project_inputs = False

                    dense_layer = DenseResidualLayer(
                        training=training,
                        units=n_timesteps * units_encoder[i],
                        layers_inner=resnet_n_layers_inner,
                        activation_inner=inner_activation,
                        activation=inner_activation,
                        project_inputs=project_inputs,
                        batch_normalization_decay=batch_normalization_decay,
                        session=session
                    )
                else:
                    dense_layer = DenseLayer(
                        training=training,
                        units=n_timesteps * units_encoder[i],
                        activation=inner_activation,
                        batch_normalization_decay=batch_normalization_decay,
                        session=session
                    )

                lambdas.append(make_lambda(dense_layer, session=None))

            dense_layer = DenseLayer(
                training=training,
                units=units_encoder[-1],
                activation=activation,
                batch_normalization_decay=batch_normalization_decay,
                session=session
            )

            lambdas.append(make_lambda(dense_layer, session=None))

            out = compose_lambdas(lambdas, **kwargs)

            return out


def rnn_decoder(
        input_shape,
        units_decoder,
        n_timesteps,
        training=True,
        add_timestep_indices=False,
        inner_activation='tanh',
        recurrent_activation='tanh',
        activation='tanh',
        batch_normalization_decay=None,
        session=None
):
    session = get_session(session)
    n_layers = len(units_decoder)
    lambdas = []
    kwargs = {'mask': None}

    with session.as_default():
        with session.graph.as_default():
            tile_dims = [1] * (len(input_shape) + 1)
            tile_dims[-2] = n_timesteps

            def make_input_layer(tile_dims, n_timesteps, add_timestep_indices):
                if add_timestep_indices:
                    def input_layer(x, mask=None):
                        out = tf.tile(
                            x[..., None, :],
                            tile_dims
                        )

                        time_ix = tf.range(n_timesteps)
                        time_ix_tile_dims = [1, 1]
                        i = -3
                        while len(time_ix.shape) < len(out.shape):
                            time_ix = time_ix[None, :]
                            time_ix_tile_dims.append(tf.shape(out)[i])
                            i -= 1

                        time_feat = tf.one_hot(time_ix, n_timesteps)
                        time_feat = tf.tile(
                            time_feat,
                            time_ix_tile_dims
                        )

                        out = tf.concat(
                            [out, time_feat],
                            axis=-1
                        )

                        if mask is not None:
                            out *= mask

                        return out

                    return input_layer

                else:
                    def input_layer(x, mask=None):
                        out = tf.tile(
                            x[..., None, :],
                            tile_dims
                        )

                        if mask is not None:
                            out *= mask

                        return out

                    return input_layer

            tiling_layer = make_input_layer(tile_dims, n_timesteps, add_timestep_indices)
            lambdas.append(make_lambda(tiling_layer, session=None, use_kwargs=True))

            multi_rnn_layer = MultiRNNLayer(
                units=units_decoder,
                layers=n_layers,
                activation=inner_activation,
                inner_activation=inner_activation,
                recurrent_activation=recurrent_activation,
                refeed_outputs=True,
                return_sequences=True,
                name='RNNDecoder',
                session=session
            )

            lambdas.append(make_lambda(multi_rnn_layer, session=None, use_kwargs=True))

            decoder = DenseLayer(
                training=training,
                units=units_decoder[-1],
                activation=activation,
                batch_normalization_decay=batch_normalization_decay,
                session=session
            )

            def final_decoder_layer(x, mask=None):
                out = decoder(x)
                if mask is not None:
                    out *= mask
                return out

            lambdas.append(make_lambda(final_decoder_layer, session=None, use_kwargs=True))

            out = compose_lambdas(lambdas, **kwargs)

            return out


