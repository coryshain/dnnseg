import re
import collections
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


def padded_boolean_mask(X, mask, session=None):
    '''
    Apply boolean mask along time dimension, with padding to keep output dimension constant.

    :param X: 3-D tensor, shape = [B, T, K] (batch, time, feature). Input tensor to mask.
    :param mask: 2-D tensor, shape = [B, T] (batch, time). Mask tensor
    :param session: session. If ``None``, uses default in current namespace.
    :return: 3-D tensor, same shape as X. Masked timesteps in each batch entry are removed so that unmasked timesteps are adjacent, with trailing zeros to keep dimensionality unchanged.
    '''

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            ix = tf.cast(tf.cumsum(mask, axis=-1) - 1, dtype=tf.int32)
            loc = tf.cast(tf.where(mask), dtype=tf.int32)
            gather_ix = loc
            gathered_X = tf.gather_nd(X, gather_ix)
            gathered_mask = tf.gather_nd(mask, gather_ix)

            gathered_ix = tf.gather_nd(ix, loc)
            scatter_ix = tf.stack([loc[:, 0], gathered_ix], axis=1)
            scattered_X = tf.scatter_nd(scatter_ix, gathered_X, tf.shape(X))
            scattered_mask = tf.scatter_nd(scatter_ix, gathered_mask, tf.shape(mask))

            return scattered_X, scattered_mask


def mask_and_lag(X, mask, n_forward=0, n_backward=0, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            X_src = tf.boolean_mask(X, mask)
            pad_base = [(0,0) for _ in range(len(X_src.shape)-2)]

            time_ix = tf.range(tf.shape(X)[-2])
            tile_ix = [1]
            for i in range(len(mask.shape) - len(time_ix.shape) - 1, -1, -1):
                tile_ix.insert(0, tf.shape(mask)[i])
                time_ix = time_ix[None,...]
            time_ix = tf.tile(time_ix, tile_ix)

            n_mask = tf.cast(tf.reduce_sum(mask, axis=-1, keepdims=True), dtype=tf.int32)
            time_mask = time_ix < n_mask

            _X_bwd = []
            _X_fwd = []
            _X_mask_bwd = []
            _X_mask_fwd = []

            for i in range(n_backward, 0, -1):
                _pad_left = tf.minimum(i, tf.shape(X_src)[-2])
                _X_cur = tf.pad(X_src[...,:-i,:], pad_base + [(_pad_left,0), (0,0)])
                _X_bwd.append(_X_cur)

                _X_mask_cur = tf.cast(time_ix >= i, dtype=mask.dtype)
                _X_mask_cur = tf.boolean_mask(_X_mask_cur, time_mask)
                _X_mask_bwd.append(_X_mask_cur)

            for i in range(1,n_forward+1):
                _pad_right = tf.minimum(i, tf.shape(X_src)[-2])
                _X_cur = tf.pad(X_src[...,i:,:], pad_base + [(0,_pad_right), (0,0)])
                _X_fwd.append(_X_cur)

                _X_mask_cur = tf.cast(time_ix < (n_mask - i), dtype=mask.dtype)
                _X_mask_cur = tf.boolean_mask(_X_mask_cur, time_mask)
                _X_mask_fwd.append(_X_mask_cur)

            if n_backward:
                _X_bwd = tf.stack(_X_bwd, axis=-2)
                _X_mask_bwd = tf.stack(_X_mask_bwd, axis=-1)
            if n_forward:
                _X_fwd = tf.stack(_X_fwd, axis=-2)
                _X_mask_fwd = tf.stack(_X_mask_fwd, axis=-1)

            return _X_bwd, _X_mask_bwd, _X_fwd, _X_mask_fwd

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


def make_clipped_linear_activation(lb=None, ub=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if lb is None:
                lb = -np.inf
            if ub is None:
                ub = np.inf
            return lambda x: tf.clip_by_value(x, lb, ub)


def get_activation(activation, session=None, training=True, from_logits=True, sample_at_train=True, sample_at_eval=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            hard_sigmoid = tf.keras.backend.hard_sigmoid

            if activation:
                if isinstance(activation, str):
                    if activation.lower().startswith('cla'):
                        _, lb, ub = activation.split('_')
                        if lb in ['None', '-inf']:
                            lb = None
                        else:
                            lb = float(lb)
                        if ub in ['None', 'inf']:
                            ub = None
                        else:
                            ub = float(ub)
                        out = make_clipped_linear_activation(lb=lb, ub=ub, session=session)
                    elif activation.lower() == 'hard_sigmoid':
                        out = hard_sigmoid
                    elif activation.lower() == 'bsn':
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

                    elif activation.lower().startswith('slow_sigmoid'):
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

                    else:
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

                tf.keras.initializers.he_normal()

                if 'identity' in initializer_name:
                    return tf.keras.initializers.Identity
                elif 'he_' in initializer_name:
                    return tf.keras.initializers.VarianceScaling(scale=2., mode='fan_in', distribution='normal')
                else:
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


def initialize_embeddings(categories, dim, default=0., name=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            categories = sorted(list(set(categories)))
            n_categories = len(categories)
            index_table = tf.contrib.lookup.index_table_from_tensor(
                tf.constant(categories),
                num_oov_buckets=1
            )
            embedding_matrix = tf.Variable(tf.fill([n_categories+1, dim], default), name=name)

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


def compose_lambdas(lambdas):
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


HMLSTM_RETURN_SIGNATURE = [
    'c',
    'h',
    'z_prob',
    'z',
    'lm',
    'cell_proposal',
    'u',
    'v',
    'w'
]


def hmlstm_state_size(
        units,
        layers,
        units_below=None,
        use_timing_unit=False,
        return_c=True,
        return_z_prob=True,
        return_lm=True,
        return_cell_proposal=True,
        return_cae=True
):
    if return_lm:
        assert units_below is not None, 'units_below must be provided when using return_lm'
    if return_cae:
        assert units_below is not None, 'units_below must be provided when using return_averaged_inputs'
    size = []
    if isinstance(units, tuple):
        units = list(units)
    if not isinstance(units, list):
        units = [units] * layers
    if units_below is not None:
        if isinstance(units_below, tuple):
            units_below = list(units_below)
        if not isinstance(units_below, list):
            units_below = [units_below] * layers
    for l in range(layers):
        size_cur = []
        size_names_cur = []

        for name in HMLSTM_RETURN_SIGNATURE:
            include = False
            value = None
            if name == 'c' and return_c:
                include = True
                value = units[l]
            elif name == 'h':
                include = True
                if use_timing_unit:
                    value = units[l] + 1
                else:
                    value = units[l]
            elif name in ['z', 'z_prob']:
                if l < layers - 1 and (name == 'z' or return_z_prob):
                    include = True
                    value = 1
            elif name == 'lm':
                if return_lm:
                    include = True
                    value = units_below[l]
            elif name == 'cell_proposal':
                if return_cell_proposal:
                    include = True
                    value = units[l]
            elif name in ['u', 'v', 'w']:
                if return_cae and l < layers - 1:
                    include = True
                    if name == 'u':
                        value = 1
                    else:
                        value = units_below[l]
            else:
                raise ValueError('No case defined for HMLSTM_RETURN_SIGNATURE item "%s".' % name)

            if include:
                size_names_cur.append(name)
                size_cur.append(value)

        state_tuple = collections.namedtuple('HMLSTMStateTupleL%d' % l, ' '.join(size_names_cur))
        size_cur = state_tuple(*size_cur)

        size.append(size_cur)

    return size


class HMLSTMCell(LayerRNNCell):
    def __init__(
            self,
            num_units,
            num_layers,
            training=False,
            kernel_depth=2,
            resnet_n_layers=2,
            one_hot_inputs=False,
            forget_bias=1.0,
            oracle_boundary=False,
            infer_boundary=True,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            boundary_activation='sigmoid',
            boundary_noise_sd=None,
            boundary_discretizer=None,
            bottomup_initializer='he_normal_initializer',
            recurrent_initializer='identity_initializer',
            topdown_initializer='he_normal_initializer',
            boundary_initializer='orthogonal_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            topdown_regularizer=None,
            boundary_regularizer=None,
            bias_regularizer=None,
            temporal_dropout=None,
            temporal_dropout_plug_lm=False,
            return_cae=True,
            return_lm_predictions=False,
            lm_order_bwd=0,
            lm_order_fwd=1,
            bottomup_dropout=None,
            recurrent_dropout=None,
            topdown_dropout=None,
            boundary_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            refeed_boundary=False,
            power=None,
            use_timing_unit=False,
            use_bias=True,
            boundary_slope_annealing_rate=None,
            state_slope_annealing_rate=None,
            slope_annealing_max=None,
            state_discretizer=None,
            discretize_state_at_boundary=False,
            nested_boundaries=False,
            state_noise_sd=None,
            sample_at_train=True,
            sample_at_eval=False,
            global_step=None,
            implementation=1,
            batch_normalization_decay=None,
            decoder_embedding=None,
            bptt=True,
            reuse=None,
            name=None,
            dtype=None,
            session=None,
            device=None
    ):
        self._session = get_session(session)
        self._device = device

        assert not temporal_dropout_plug_lm, 'temporal_dropout_plug_lm is currently broken, do not use.'

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

                    self._kernel_depth = kernel_depth
                    self._resnet_n_layers = resnet_n_layers

                    self._return_cae = return_cae
                    self._lm = return_lm_predictions or temporal_dropout_plug_lm
                    self._return_lm_predictions = return_lm_predictions
                    self._lm_order_bwd = lm_order_bwd
                    self._lm_order_fwd = lm_order_fwd
                    self._temporal_dropout_plug_lm = temporal_dropout_plug_lm
                    self._one_hot_inputs = one_hot_inputs

                    self._forget_bias = forget_bias

                    self._oracle_boundary = oracle_boundary
                    self._infer_boundary = infer_boundary

                    self._activation = get_activation(activation, session=self._session, training=self._training)
                    self._inner_activation = get_activation(inner_activation, session=self._session, training=self._training)
                    # self._prefinal_activation = self._inner_activation
                    self._prefinal_activation = tf.nn.elu
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
                    self.boundary_noise_sd = boundary_noise_sd

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
                    self._use_timing_unit = use_timing_unit
                    self._use_bias = use_bias
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
                    self._discretize_state_at_boundary = discretize_state_at_boundary
                    self._nested_boundaries = nested_boundaries
                    self.state_noise_sd = state_noise_sd
                    self._sample_at_train = sample_at_train
                    self._sample_at_eval = sample_at_eval
                    self.global_step = global_step
                    self._implementation = implementation

                    self._batch_normalization_decay = batch_normalization_decay

                    self._decoder_embedding = decoder_embedding

                    self._bptt = bptt

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
        units_below = [self._input_dims] + self._num_units[:-1]
        for l in range(len(units_below)):
            units_below[l] = units_below[l] * (self._lm_order_bwd + self._lm_order_fwd)

        return hmlstm_state_size(
            self._num_units,
            self._num_layers,
            units_below=units_below,
            use_timing_unit=self._use_timing_unit,
            return_lm = self._lm,
            return_c=True,
            return_z_prob=True,
            return_cell_proposal=True,
            return_cae=self._return_cae
        )

    @property
    def output_size(self):
        return self.state_size

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
                    if self._oracle_boundary:
                        self._input_dims -= self._num_layers - 1

                    self._kernel_bottomup = []
                    self._kernel_recurrent = []
                    self._kernel_topdown = []
                    if self._use_bias:
                        self._bias_boundary = []
                    if self._lm:
                        self._kernel_lm_bottomup = []
                        self._kernel_lm_recurrent = []
                        self._lm_output_gain = []
                    if self._return_cae:
                        self._kernel_w = []

                    if self._use_bias:
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
                            bottom_up_dim = self._num_units[l - 1] + self._use_timing_unit
                        bottom_up_dim += self._refeed_boundary and (self._implementation == 1)

                        recurrent_dim = self._num_units[l] + (self._use_timing_unit and l < self._num_layers - 1)

                        if self._implementation == 1:
                            if l < self._num_layers - 1:
                                output_dim = 4 * self._num_units[l] + 1
                            else:
                                output_dim = 4 * self._num_units[l]
                        else:
                            output_dim = 4 * self._num_units[l]

                        kernel_bottomup_lambdas = []
                        for d in range(self._kernel_depth):
                            if d < self._kernel_depth - 1:
                                if self._resnet_n_layers > 1:
                                    kernel_bottomup_layer = DenseResidualLayer(
                                        training=self._training,
                                        units=bottom_up_dim,
                                        use_bias=self._use_bias,
                                        kernel_initializer='identity_initializer',
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._bottomup_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        layers_inner=self._resnet_n_layers,
                                        activation_inner=self._prefinal_activation,
                                        activation=self._prefinal_activation,
                                        batch_normalization_decay=self._batch_normalization_decay,
                                        project_inputs=False,
                                        normalize_weights=self._weight_normalization,
                                        reuse=tf.AUTO_REUSE,
                                        session=self._session,
                                        name='kernel_bottomup_l%d_d%d' % (l, d)
                                    )
                                else:
                                    kernel_bottomup_layer = DenseLayer(
                                        training=self._training,
                                        units=bottom_up_dim,
                                        use_bias=self._use_bias,
                                        kernel_initializer='identity_initializer',
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._bottomup_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        activation=self._prefinal_activation,
                                        batch_normalization_decay=self._batch_normalization_decay,
                                        normalize_weights=self._weight_normalization,
                                        session=self._session,
                                        reuse=tf.AUTO_REUSE,
                                        name='kernel_bottomup_l%d_d%d' % (l, d)
                                    )
                            else:
                                kernel_bottomup_layer = DenseLayer(
                                    training=self._training,
                                    units=output_dim,
                                    use_bias=self._use_bias,
                                    kernel_initializer=self._bottomup_initializer,
                                    bias_initializer=self._bias_initializer,
                                    kernel_regularizer=self._bottomup_regularizer,
                                    bias_regularizer=self._bias_regularizer,
                                    activation=None,
                                    batch_normalization_decay=self._batch_normalization_decay,
                                    normalize_weights=self._weight_normalization,
                                    session=self._session,
                                    reuse=tf.AUTO_REUSE,
                                    name='kernel_bottomup_l%d_d%d' % (l, d)
                                )
                            kernel_bottomup_lambdas.append(make_lambda(kernel_bottomup_layer, session=self._session))

                        kernel_bottomup = compose_lambdas(kernel_bottomup_lambdas)

                        self._kernel_bottomup.append(kernel_bottomup)

                        kernel_recurrent_lambdas = []
                        for d in range(self._kernel_depth):
                            if d < self._kernel_depth - 1:
                                if self._resnet_n_layers > 1:
                                    kernel_recurrent_layer = DenseResidualLayer(
                                        training=self._training,
                                        units=recurrent_dim,
                                        use_bias=self._use_bias,
                                        kernel_initializer='identity_initializer',
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._recurrent_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        layers_inner=self._resnet_n_layers,
                                        activation_inner=self._prefinal_activation,
                                        activation=self._prefinal_activation,
                                        batch_normalization_decay=self._batch_normalization_decay,
                                        project_inputs=False,
                                        normalize_weights=self._weight_normalization,
                                        reuse=tf.AUTO_REUSE,
                                        session=self._session,
                                        name='kernel_recurrent_l%d_d%d' % (l, d)
                                    )
                                else:
                                    kernel_recurrent_layer = DenseLayer(
                                        training=self._training,
                                        units=recurrent_dim,
                                        use_bias=self._use_bias,
                                        kernel_initializer='identity_initializer',
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._recurrent_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        activation=self._prefinal_activation,
                                        batch_normalization_decay=self._batch_normalization_decay,
                                        normalize_weights=self._weight_normalization,
                                        session=self._session,
                                        reuse=tf.AUTO_REUSE,
                                        name='kernel_recurrent_l%d_d%d' % (l, d)
                                    )
                            else:
                                kernel_recurrent_layer = DenseLayer(
                                    training=self._training,
                                    units=output_dim,
                                    use_bias=self._use_bias,
                                    kernel_initializer=self._recurrent_initializer,
                                    bias_initializer=self._bias_initializer,
                                    kernel_regularizer=self._recurrent_regularizer,
                                    bias_regularizer=self._bias_regularizer,
                                    activation=None,
                                    batch_normalization_decay=self._batch_normalization_decay,
                                    normalize_weights=self._weight_normalization,
                                    session=self._session,
                                    reuse=tf.AUTO_REUSE,
                                    name='kernel_recurrent_l%d_d%d' % (l, d)
                                )
                            kernel_recurrent_lambdas.append(make_lambda(kernel_recurrent_layer, session=self._session))

                        kernel_recurrent = compose_lambdas(kernel_recurrent_lambdas)

                        self._kernel_recurrent.append(kernel_recurrent)

                        if l < self._num_layers - 1:
                            top_down_dim = self._num_units[l + 1] + (self._use_timing_unit and l < self._num_layers - 2)

                            kernel_topdown_lambdas = []
                            for d in range(self._kernel_depth):
                                if d < self._kernel_depth - 1:
                                    if self._resnet_n_layers < 1:
                                        kernel_topdown_layer = DenseResidualLayer(
                                            training=self._training,
                                            units=top_down_dim,
                                            use_bias=self._use_bias,
                                            kernel_initializer='identity_initializer',
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._topdown_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            layers_inner=self._resnet_n_layers,
                                            activation_inner=self._prefinal_activation,
                                            activation=self._prefinal_activation,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            project_inputs=False,
                                            normalize_weights=self._weight_normalization,
                                            reuse=tf.AUTO_REUSE,
                                            session=self._session,
                                            name='kernel_topdown_l%d_d%d' % (l, d)
                                        )
                                    else:
                                        kernel_topdown_layer = DenseLayer(
                                            training=self._training,
                                            units=top_down_dim,
                                            use_bias=self._use_bias,
                                            kernel_initializer='identity_initializer',
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._topdown_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            activation=self._prefinal_activation,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            normalize_weights=self._weight_normalization,
                                            session=self._session,
                                            reuse=tf.AUTO_REUSE,
                                            name='kernel_topdown_l%d_d%d' % (l, d)
                                        )
                                else:
                                    kernel_topdown_layer = DenseLayer(
                                        training=self._training,
                                        units=output_dim,
                                        use_bias=self._use_bias,
                                        kernel_initializer=self._topdown_initializer,
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._topdown_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        activation=None,
                                        batch_normalization_decay=self._batch_normalization_decay,
                                        normalize_weights=self._weight_normalization,
                                        session=self._session,
                                        reuse=tf.AUTO_REUSE,
                                        name='kernel_topdown_l%d_d%d' % (l, d)
                                    )
                                kernel_topdown_lambdas.append(make_lambda(kernel_topdown_layer, session=self._session))

                            kernel_topdown = compose_lambdas(kernel_topdown_lambdas)

                            self._kernel_topdown.append(kernel_topdown)

                        if self._lm:
                            self._lm_output_gain.append(
                                tf.nn.softplus(
                                    self.add_variable(
                                        'lm_output_gain_l%d' % l,
                                        initializer=tf.constant_initializer(np.log(np.e - 1)),
                                        shape=[]
                                    )
                                )
                                # tf.ones(shape=[])
                            )

                            _kernel_depth = self._kernel_depth
                            # _kernel_depth = 1

                            lm_out_dim = (bottom_up_dim - (self._use_timing_unit and l > 0)) * (self._lm_order_bwd + self._lm_order_fwd)

                            lm_bottomup_in_dim = recurrent_dim # intentional
                            if l == 0 and self._decoder_embedding is not None:
                                lm_bottomup_in_dim += int(self._decoder_embedding.shape[-1])
                            kernel_lm_bottomup_lambdas = []
                            for d in range(_kernel_depth):
                                if d < _kernel_depth - 1:
                                    if self._resnet_n_layers > 1:
                                        kernel_lm_bottomup_layer = DenseResidualLayer(
                                            training=self._training,
                                            units=lm_bottomup_in_dim,
                                            use_bias=self._use_bias,
                                            kernel_initializer='identity_initializer',
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._bottomup_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            layers_inner=self._resnet_n_layers,
                                            activation_inner=self._prefinal_activation,
                                            activation=self._prefinal_activation,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            project_inputs=False,
                                            normalize_weights=self._weight_normalization,
                                            reuse=tf.AUTO_REUSE,
                                            session=self._session,
                                            name='kernel_lm_bottomup_l%d_d%d' % (l, d)
                                        )
                                    else:
                                        kernel_lm_bottomup_layer = DenseLayer(
                                            training=self._training,
                                            units=lm_bottomup_in_dim,
                                            use_bias=self._use_bias,
                                            kernel_initializer='identity_initializer',
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._bottomup_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            activation=self._prefinal_activation,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            normalize_weights=self._weight_normalization,
                                            session=self._session,
                                            reuse=tf.AUTO_REUSE,
                                            name='kernel_lm_bottomup_l%d_d%d' % (l, d)
                                        )
                                else:
                                    kernel_lm_bottomup_layer = DenseLayer(
                                        training=self._training,
                                        units=lm_out_dim,
                                        use_bias=True,
                                        kernel_initializer=self._bottomup_initializer,
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._bottomup_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        activation=None,
                                        batch_normalization_decay=self._batch_normalization_decay,
                                        normalize_weights=self._weight_normalization,
                                        session=self._session,
                                        reuse=tf.AUTO_REUSE,
                                        name='kernel_lm_bottomup_l%d_d%d' % (l, d)
                                    )
                                kernel_lm_bottomup_lambdas.append(make_lambda(kernel_lm_bottomup_layer, session=self._session))

                            kernel_lm_bottomup = compose_lambdas(kernel_lm_bottomup_lambdas)

                            self._kernel_lm_bottomup.append(kernel_lm_bottomup)

                            lm_recurrent_in_dim = bottom_up_dim # intentional
                            if l == 0 and self._decoder_embedding is not None:
                                lm_recurrent_in_dim += int(self._decoder_embedding.shape[-1])
                            kernel_lm_recurrent_lambdas = []
                            for d in range(_kernel_depth):
                                if d < _kernel_depth - 1:
                                    if self._resnet_n_layers > 1:
                                        kernel_lm_recurrent_layer = DenseResidualLayer(
                                            training=self._training,
                                            units=lm_recurrent_in_dim,
                                            use_bias=self._use_bias,
                                            kernel_initializer='identity_initializer',
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._recurrent_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            layers_inner=self._resnet_n_layers,
                                            activation_inner=self._prefinal_activation,
                                            activation=self._prefinal_activation,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            project_inputs=False,
                                            normalize_weights=self._weight_normalization,
                                            reuse=tf.AUTO_REUSE,
                                            session=self._session,
                                            name='kernel_lm_recurrent_l%d_d%d' % (l, d)
                                        )
                                    else:
                                        kernel_lm_recurrent_layer = DenseLayer(
                                            training=self._training,
                                            units=lm_recurrent_in_dim,
                                            use_bias=self._use_bias,
                                            kernel_initializer='identity_initializer',
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._recurrent_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            activation=self._prefinal_activation,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            normalize_weights=self._weight_normalization,
                                            session=self._session,
                                            reuse=tf.AUTO_REUSE,
                                            name='kernel_lm_recurrent_l%d_d%d' % (l, d)
                                        )
                                else:
                                    kernel_lm_recurrent_layer = DenseLayer(
                                        training=self._training,
                                        units=lm_out_dim,
                                        use_bias=True,
                                        kernel_initializer=self._recurrent_initializer,
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._recurrent_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        activation=None,
                                        batch_normalization_decay=self._batch_normalization_decay,
                                        normalize_weights=self._weight_normalization,
                                        session=self._session,
                                        reuse=tf.AUTO_REUSE,
                                        name='kernel_lm_recurrent_l%d_d%d' % (l, d)
                                    )
                                kernel_lm_recurrent_lambdas.append(make_lambda(kernel_lm_recurrent_layer, session=self._session))

                            kernel_lm_recurrent = compose_lambdas(kernel_lm_recurrent_lambdas)

                            self._kernel_lm_recurrent.append(kernel_lm_recurrent)

                        if self._return_cae:
                            _kernel_depth = self._kernel_depth
                            w_out_dim = bottom_up_dim - (self._use_timing_unit and l > 0)
                            w_in_dim = recurrent_dim
                            if l == 0 and self._decoder_embedding is not None:
                                w_in_dim += int(self._decoder_embedding.shape[-1])
                            kernel_w_lambdas = []
                            for d in range(_kernel_depth):
                                if d < _kernel_depth - 1:
                                    if self._resnet_n_layers > 1:
                                        kernel_w_layer = DenseResidualLayer(
                                            training=self._training,
                                            units=w_in_dim,
                                            use_bias=self._use_bias,
                                            kernel_initializer='identity_initializer',
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._bottomup_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            layers_inner=self._resnet_n_layers,
                                            activation_inner=self._prefinal_activation,
                                            activation=self._prefinal_activation,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            project_inputs=False,
                                            normalize_weights=self._weight_normalization,
                                            reuse=tf.AUTO_REUSE,
                                            session=self._session,
                                            name='kernel_w_l%d_d%d' % (l, d)
                                        )
                                    else:
                                        kernel_w_layer = DenseLayer(
                                            training=self._training,
                                            units=w_in_dim,
                                            use_bias=self._use_bias,
                                            kernel_initializer='identity_initializer',
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._bottomup_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            activation=self._prefinal_activation,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            normalize_weights=self._weight_normalization,
                                            session=self._session,
                                            reuse=tf.AUTO_REUSE,
                                            name='kernel_w_l%d_d%d' % (l, d)
                                        )
                                else:
                                    kernel_w_layer = DenseLayer(
                                        training=self._training,
                                        units=w_out_dim,
                                        use_bias=True,
                                        kernel_initializer=self._bottomup_initializer,
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._bottomup_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        activation=None,
                                        batch_normalization_decay=self._batch_normalization_decay,
                                        normalize_weights=self._weight_normalization,
                                        session=self._session,
                                        reuse=tf.AUTO_REUSE,
                                        name='kernel_w_l%d_d%d' % (l, d)
                                    )
                                kernel_w_lambdas.append(
                                    make_lambda(kernel_w_layer, session=self._session))

                            kernel_w_lambdas = compose_lambdas(kernel_w_lambdas)

                            self._kernel_w.append(kernel_w_lambdas)

                        if not self._layer_normalization and self._use_bias:
                            bias = self.add_variable(
                                'bias_%d' % l,
                                shape=[1, output_dim],
                                initializer=self._bias_initializer
                            )
                            self._regularize(bias, self._bias_regularizer)
                            self._bias.append(bias)

                        if l < self._num_layers - 1 and self._infer_boundary:
                            if self._implementation == 1 and not self._layer_normalization and self._use_bias:
                                bias_boundary = bias[:, -1:]
                                self._bias_boundary.append(bias_boundary)

                            if self._implementation == 2:
                                kernel_boundary_lambdas = []
                                for d in range(self._kernel_depth):
                                    if d < self._kernel_depth - 1:
                                        if self._resnet_n_layers > 1:
                                            kernel_boundary_layer = DenseResidualLayer(
                                                training=self._training,
                                                units=self._num_units[l] + self._use_timing_unit,
                                                use_bias=self._use_bias,
                                                kernel_initializer='identity_initializer',
                                                bias_initializer=self._bias_initializer,
                                                kernel_regularizer=self._boundary_regularizer,
                                                bias_regularizer=self._bias_regularizer,
                                                layers_inner=self._resnet_n_layers,
                                                activation_inner=self._prefinal_activation,
                                                activation=self._prefinal_activation,
                                                batch_normalization_decay=self._batch_normalization_decay,
                                                project_inputs=False,
                                                normalize_weights=self._weight_normalization,
                                                reuse=tf.AUTO_REUSE,
                                                session=self._session,
                                                name='kernel_boundary_l%d_d%d' % (l, d)
                                            )
                                        else:
                                            kernel_boundary_layer = DenseLayer(
                                                training=self._training,
                                                units=self._num_units[l] + self._use_timing_unit,
                                                use_bias=self._use_bias,
                                                kernel_initializer='identity_initializer',
                                                bias_initializer=self._bias_initializer,
                                                kernel_regularizer=self._boundary_regularizer,
                                                bias_regularizer=self._bias_regularizer,
                                                activation=self._prefinal_activation,
                                                batch_normalization_decay=self._batch_normalization_decay,
                                                normalize_weights=self._weight_normalization,
                                                session=self._session,
                                                reuse=tf.AUTO_REUSE,
                                                name='kernel_boundary_l%d_d%d' % (l, d)
                                            )
                                    else:
                                        kernel_boundary_layer = DenseLayer(
                                            training=self._training,
                                            units=1,
                                            use_bias=self._use_bias,
                                            kernel_initializer=self._boundary_initializer,
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._boundary_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            activation=None,
                                            batch_normalization_decay=self._batch_normalization_decay,
                                            normalize_weights=self._weight_normalization,
                                            session=self._session,
                                            reuse=tf.AUTO_REUSE,
                                            name='kernel_boundary_l%d_d%d' % (l, d)
                                        )
                                    kernel_boundary_lambdas.append(make_lambda(kernel_boundary_layer, session=self._session))

                                kernel_boundary = compose_lambdas(kernel_boundary_lambdas)

                                # kernel_boundary = self.add_variable(
                                #     'kernel_boundary_%d' % l,
                                #     shape=[self._num_units[l] + self._refeed_boundary, 1],
                                #     initializer=self._boundary_initializer
                                # )
                                # if self._weight_normalization:
                                #     kernel_boundary_g = tf.Variable(tf.ones([1, output_dim]), name='kernel_boundary_g_%d' % l)
                                #     kernel_boundary_b = tf.Variable(tf.zeros([1, output_dim]), name='kernel_boundary_b_%d' % l)
                                #     kernel_boundary = tf.nn.l2_normalize(kernel_boundary, axis=-1) * kernel_boundary_g + kernel_boundary_b
                                # self._regularize(kernel_boundary, self._boundary_regularizer)

                                self._kernel_boundary.append(kernel_boundary)

                                if self._use_bias:
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

                        if not self._bptt:
                            h_below = tf.stop_gradient(h_below)
                            # if z_below is not None:
                            #     print('Stopping gradient z_below')
                            #     z_below = tf.stop_gradient(z_below)

                        timing = int(self._use_timing_unit and l < self._num_layers - 1)

                        # z_behind: Previous boundary probability at current layer (implicitly 1 if final layer)
                        if l < self._num_layers - 1:
                            z_behind = layer.z
                            # print('Stopping gradient z_behind')
                            # z_behind = tf.stop_gradient(z_behind)
                            if self._return_cae:
                                u = layer.u
                                v = layer.v
                            else:
                                u = None
                                v = None
                        else:
                            z_behind = None
                            u = None
                            v = None

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
                        c_behind = layer.c
                        if not self._bptt:
                            c_behind = tf.stop_gradient(c_behind)

                        # h_behind: Previous hidden state at current layer
                        h_behind = layer.h
                        if not self._bptt:
                            h_behind = tf.stop_gradient(h_behind)
                        h_behind = self._recurrent_dropout(h_behind)

                        # h_above: Hidden state of layer above at previous timestep (implicitly 0 if final layer)
                        if l < self._num_layers - 1:
                            h_above = state[l + 1].h
                            if not self._bptt:
                                h_above = tf.stop_gradient(h_above)
                            h_above = self._topdown_dropout(h_above)
                        else:
                            h_above = None

                        if self._implementation == 1 and self._refeed_boundary:
                            h_below = tf.concat([z_behind, h_below], axis=1)

                        # h_below = self._temporal_dropout[l](h_below)
                        if l > 0 and self._state_discretizer and self._discretize_state_at_boundary and self.global_step is not None:
                            h_below = self._state_discretizer(h_below)
                        h_below = self._bottomup_dropout(h_below)

                        s_bottomup = self._kernel_bottomup[l](h_below)
                        if l > 0:
                            s_bottomup *= z_below ** power

                        if self._temporal_dropout[l]:
                            def train_func():
                                bottomup_features = s_bottomup
                                noise = tf.random_uniform([tf.shape(bottomup_features)[0]]) > self._temporal_dropout[l]
                                alt = tf.zeros_like(bottomup_features)
                                out = tf.where(noise, bottomup_features, alt)
                                return out, tf.cast(noise, dtype=tf.float32)[..., None]

                            def eval_func():
                                return s_bottomup, tf.ones([tf.shape(s_bottomup)[0], 1])

                            s_bottomup, normalizer = tf.cond(self._training, train_func, eval_func)
                        else:
                            normalizer = 1.
                        if l > 0:
                            normalizer *= z_below

                        # Recurrent features
                        s_recurrent = self._kernel_recurrent[l](h_behind)
                        normalizer += 1.

                        # Sum bottom-up and recurrent features
                        s = s_bottomup + s_recurrent

                        # Top-down features (if non-final layer)
                        if l < self._num_layers - 1:
                            # Compute top-down features
                            s_topdown = self._kernel_topdown[l](h_above) * (z_behind ** power)
                            normalizer += z_behind
                            # Add in top-down features
                            s = s + s_topdown

                        # normalizer = tf.Print(normalizer, ['l%d_normalizer' % l, normalizer], summarize=32)

                        s /= normalizer

                        if self._implementation == 1:
                            # In implementation 1, boundary has its own slice of the hidden state preactivations
                            z_logit = s[:, units * 4:]
                            s = s[:, :units * 4]
                        else:
                            z_logit = None

                        if not self._layer_normalization and self._use_bias:
                            if self._implementation == 1:
                                z_logit += self._bias[l][:, units * 4:]
                            s = s + self._bias[l][:, :units * 4]

                        # Forget gate
                        f = s[:, :units]
                        if self._layer_normalization:
                            f = self.norm(f, 'f_ln_%d' % l)
                        f = self._recurrent_activation(f + self._forget_bias)

                        # Input gate
                        i = s[:, units:units * 2]
                        if self._layer_normalization:
                            i = self.norm(i, 'i_ln_%d' % l)
                        i = self._recurrent_activation(i)

                        # Output gate
                        o = s[:, units * 2:units * 3]
                        if self._layer_normalization:
                            o = self.norm(o, 'o_ln_%d' % l)
                        if self.state_noise_sd and self._state_discretizer:
                            # h_in = tf.cond(self._training, lambda: c + tf.random_normal(shape=tf.shape(c), stddev=self.state_noise_sd), lambda: c)
                            output_gate_noise = tf.random_normal(shape=tf.shape(o), stddev=self.state_noise_sd)
                            o += output_gate_noise
                        o = self._recurrent_activation(o)

                        # Cell proposal
                        g = s[:, units * 3:units * 4]
                        if self._layer_normalization:
                            g = self.norm(g, 'g_ln_%d' % l)
                        g = activation(g)

                        # Compute cell state flush operation
                        c_flush = i * g

                        # Compute cell state copy operation
                        c_copy = c_behind

                        # Cell state update (forget-gated previous cell plus input-gated cell proposal)
                        c_update = f * c_behind + c_flush

                        # Merge cell operations. If boundaries are hard, selects between update, copy, and flush.
                        # If boundaries are soft, sums update copy and flush proportionally to their probs.
                        c = update_prob * c_update + copy_prob * c_copy + flush_prob * c_flush

                        # Compute the gated output
                        # if l < self._num_layers - 1 and self._state_discretizer is not None:
                        if self._state_discretizer is not None:
                            activation = tf.sigmoid
                            # activation = tf.keras.backend.hard_sigmoid
                        if self.state_noise_sd:
                            # h_in = tf.cond(self._training, lambda: c + tf.random_normal(shape=tf.shape(c), stddev=self.state_noise_sd), lambda: c)
                            state_noise = tf.random_normal(shape=tf.shape(c), stddev=self.state_noise_sd)
                            h_in = c + state_noise
                        else:
                            h_in = c
                        if self._state_slope_annealing_rate and self.global_step is not None:
                            h = activation(h_in * self.state_slope_coef)
                        else:
                            h = activation(h_in)
                        if self._state_discretizer:
                            h *= self._state_discretizer(o)
                        else:
                            h *= o
                        # h = tf.Print(h, [h])

                        # if l < self._num_layers - 1 and self._state_discretizer and not self._discretize_state_at_boundary and self.global_step is not None:
                        if self._state_discretizer and not self._discretize_state_at_boundary and self.global_step is not None:
                            h = self._state_discretizer(h)

                        # Mix gated output with the previous hidden state proportionally to the copy probability
                        if timing:
                            h_behind_cur = h_behind[:, :-1]
                        else:
                            h_behind_cur = h_behind
                        h = copy_prob * h_behind_cur + (1 - copy_prob) * h
                        if timing:
                            timing_unit = (h_behind[:,-1:] + 0.5 * (1 - h_behind[:,-1:])) * (1 - z_behind)
                            h = tf.concat([h, timing_unit], axis=1)

                        if self._lm:
                            if l == 0 and self._decoder_embedding is not None:
                                lm_bottomup_in = tf.concat([self._decoder_embedding, h], axis=-1)
                                pred_prev = tf.concat([self._decoder_embedding, state[l].lm], axis=-1)
                            else:
                                pred_prev = state[l].lm
                                lm_bottomup_in = h
                            pred_new = self._kernel_lm_recurrent[l](pred_prev) + self._kernel_lm_bottomup[l](lm_bottomup_in)
                            pred_new = self._inner_activation(pred_new)
                            pred_new *= self._lm_output_gain[l]
                        else:
                            pred_new = None

                        # Compute the current boundary probability
                        if l < self._num_layers - 1:
                            print()
                            if self._oracle_boundary:
                                inputs_last_dim = inputs.shape[-1] - self._num_layers + 1
                                z_prob = inputs[:, inputs_last_dim + l:inputs_last_dim + l+1]
                            else:
                                z_prob = tf.zeros([tf.shape(inputs)[0], 1])
                            if self._infer_boundary:
                                z_prob_oracle = z_prob
                                if self._implementation == 2:
                                    # In implementation 2, boundary is computed by linear transform of h
                                    z_in = h
                                    if self._refeed_boundary:
                                        z_in = tf.concat([z_behind, z_in], axis=1)
                                    z_in = self._boundary_dropout(z_in)
                                    # In implementation 2, boundary is a function of the hidden state
                                    z_logit = self._kernel_boundary[l](z_in)
                                    if self._use_bias:
                                        z_logit += self._bias_boundary[l]

                                if self._boundary_slope_annealing_rate and self.global_step is not None:
                                    z_logit *= self.boundary_slope_coef

                                if self.boundary_noise_sd:
                                    z_logit = tf.cond(self._training, lambda: z_logit + tf.random_normal(shape=tf.shape(z_logit), stddev=self.boundary_noise_sd), lambda: z_logit)

                                z_prob = self._boundary_activation(z_logit)

                                if self._oracle_boundary:
                                    z_prob = tf.maximum(z_prob, z_prob_oracle)

                                if self._nested_boundaries and l > 0 and not self._boundary_discretizer:
                                    z_prob *= z_below

                                if self._boundary_discretizer:
                                    z = self._boundary_discretizer(z_prob)
                                else:
                                    z = z_prob

                                if self._nested_boundaries and l > 0 and self._boundary_discretizer:
                                    z = z * z_below
                            else:
                                z = z_prob
                        else:
                            z_prob = None
                            z = None

                        if self._return_cae and l < self._num_layers - 1:
                            v = (v * u + h_below) / (u + 1)
                            z_comp = 1 - z
                            u = u * z_comp + z_comp
                            w = self._kernel_w[l](h)
                        else:
                            v = u = w = None

                        new_state_names = []
                        new_state_cur = []
                        name_map = {
                            'c': c,
                            'h': h,
                            'z': z,
                            'z_prob': z_prob,
                            'lm': pred_new,
                            'cell_proposal': c_flush,
                            'u': u,
                            'v': v,
                            'w': w
                        }

                        for name in HMLSTM_RETURN_SIGNATURE:
                            include = False
                            if name in ['c', 'h', 'cell_proposal']:
                                include = True
                            elif name in ['z', 'z_prob'] and l < self._num_layers - 1:
                                include = True
                            elif name == 'lm' and self._lm:
                                include = True
                            elif self._return_cae and l < self._num_layers - 1 and name in ['u', 'v', 'w']:
                                include = True

                            if include:
                                new_state_names.append(name)
                                new_state_cur.append(name_map[name])

                        state_tuple = collections.namedtuple('HMLSTMStateTupleL%d' % l, ' '.join(new_state_names))
                        new_state_cur = state_tuple(*new_state_cur)

                        # Append layer to new state
                        new_state.append(new_state_cur)

                        h_below = h
                        z_below = z

                    return new_state, new_state


class HMLSTMSegmenter(object):
    def __init__(
            self,
            num_units,
            num_layers,
            training=False,
            kernel_depth=2,
            resnet_n_layers=2,
            one_hot_inputs=False,
            forget_bias=1.0,
            oracle_boundary=False,
            infer_boundary=True,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            boundary_activation='sigmoid',
            boundary_discretizer=None,
            boundary_noise_sd=None,
            bottomup_initializer='he_normal_initializer',
            recurrent_initializer='identity_initializer',
            topdown_initializer='he_normal_initializer',
            boundary_initializer='he_normal_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            topdown_regularizer=None,
            boundary_regularizer=None,
            bias_regularizer=None,
            bottomup_dropout=None,
            temporal_dropout=None,
            temporal_dropout_plug_lm=False,
            return_cae=True,
            return_lm_predictions=False,
            lm_order_bwd=0,
            lm_order_fwd=1,
            recurrent_dropout=None,
            topdown_dropout=None,
            boundary_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            refeed_boundary=False,
            power=None,
            use_timing_unit=False,
            use_bias=True,
            boundary_slope_annealing_rate=None,
            state_slope_annealing_rate=None,
            slope_annealing_max=None,
            state_discretizer=None,
            discretize_state_at_boundary=None,
            nested_boundaries=False,
            state_noise_sd=None,
            sample_at_train=True,
            sample_at_eval=False,
            global_step=None,
            implementation=1,
            decoder_embedding=None,
            bptt=True,
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

                    assert len(self.num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                    self.num_layers = num_layers
                    self.training = training
                    self.kernel_depth = kernel_depth
                    self.resnet_n_layers = resnet_n_layers
                    self.one_hot_inputs = one_hot_inputs
                    self.forget_bias = forget_bias
                    self.oracle_boundary = oracle_boundary
                    self.infer_boundary = infer_boundary

                    self.activation = activation
                    self.inner_activation = inner_activation
                    self.recurrent_activation = recurrent_activation
                    self.boundary_activation = boundary_activation
                    self.boundary_discretizer = boundary_discretizer
                    self.boundary_noise_sd = boundary_noise_sd

                    self.bottomup_initializer = bottomup_initializer
                    self.recurrent_initializer = recurrent_initializer
                    self.topdown_initializer = topdown_initializer
                    self.boundary_initializer = boundary_initializer
                    self.bias_initializer = bias_initializer

                    self.temporal_dropout = temporal_dropout
                    self.return_cae = return_cae
                    self.return_lm_predictions = return_lm_predictions
                    self.lm_order_bwd = lm_order_bwd
                    self.lm_order_fwd = lm_order_fwd
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
                    self.use_timing_unit = use_timing_unit
                    self.use_bias = use_bias
                    self.boundary_slope_annealing_rate = boundary_slope_annealing_rate
                    self.state_slope_annealing_rate = state_slope_annealing_rate
                    self.slope_annealing_max = slope_annealing_max
                    self.state_discretizer = state_discretizer
                    self.discretize_state_at_boundary = discretize_state_at_boundary
                    self.nested_boundaries = nested_boundaries
                    self.state_noise_sd = state_noise_sd
                    self.sample_at_train = sample_at_train
                    self.sample_at_eval = sample_at_eval
                    self.global_step = global_step
                    self.implementation = implementation
                    self.decoder_embedding = decoder_embedding
                    self.bptt = bptt

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
                        kernel_depth=self.kernel_depth,
                        resnet_n_layers=self.resnet_n_layers,
                        one_hot_inputs=self.one_hot_inputs,
                        forget_bias=self.forget_bias,
                        oracle_boundary=self.oracle_boundary,
                        infer_boundary=self.infer_boundary,
                        activation=self.activation,
                        inner_activation=self.inner_activation,
                        recurrent_activation=self.recurrent_activation,
                        boundary_activation=self.boundary_activation,
                        boundary_discretizer=self.boundary_discretizer,
                        boundary_noise_sd=self.boundary_noise_sd,
                        bottomup_initializer=self.bottomup_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        topdown_initializer=self.topdown_initializer,
                        bias_initializer=self.bias_initializer,
                        bottomup_regularizer=self.bottomup_regularizer,
                        recurrent_regularizer=self.recurrent_regularizer,
                        topdown_regularizer=self.topdown_regularizer,
                        boundary_regularizer=self.boundary_regularizer,
                        temporal_dropout=self.temporal_dropout,
                        return_cae=self.return_cae,
                        return_lm_predictions=self.return_lm_predictions,
                        lm_order_bwd=self.lm_order_bwd,
                        lm_order_fwd=self.lm_order_fwd,
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
                        use_timing_unit=self.use_timing_unit,
                        use_bias=self.use_bias,
                        boundary_slope_annealing_rate=self.boundary_slope_annealing_rate,
                        state_slope_annealing_rate=self.state_slope_annealing_rate,
                        slope_annealing_max=self.slope_annealing_max,
                        state_discretizer=self.state_discretizer,
                        discretize_state_at_boundary=self.discretize_state_at_boundary,
                        nested_boundaries=self.nested_boundaries,
                        state_noise_sd=self.state_noise_sd,
                        sample_at_train=self.sample_at_train,
                        sample_at_eval=self.sample_at_eval,
                        global_step=self.global_step,
                        implementation=self.implementation,
                        decoder_embedding=self.decoder_embedding,
                        bptt=self.bptt,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        session=self.session,
                        device=self.device
                    )

                    self.cell.build(inputs.shape[1:])

        self.built = True

    def __call__(self, inputs, mask=None, boundaries=None):
        assert not (self.oracle_boundary and boundaries is None), 'A boundaries arg must be provided when oracle_boundary is true'

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

    def cell(self, level=None, mask=None):
        if level is None:
            out = tuple([l.cell(mask=mask) for l in self.l])
        else:
            out = self.l[level].cell(mask=mask)

        return out

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

    def lm_logits(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                lm_logits = [l.lm_logits(mask=mask) for l in self.l]

                return lm_logits

    def cell_proposals(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                cell_proposal = [l.cell_proposals(mask=mask) for l in self.l]

                return cell_proposal

    def averaged_inputs(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                v = [l.averaged_inputs(mask=mask) for l in self.l]

                return v

    def averaged_input_logits(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                w = [l.averaged_input_logits(mask=mask) for l in self.l]

                return w

    def segment_lengths(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                u = [l.segment_lengths(mask=mask) for l in self.l]

                return u


class HMLSTMOutputLevel(object):
    def __init__(self, output, session=None):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                for prop in HMLSTM_RETURN_SIGNATURE:
                    setattr(self, prop, getattr(output, prop, None))

    def cell(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.c
                if mask is not None:
                    while len(mask.shape) < len(out.shape):
                        mask = mask[..., None]
                    out = out * mask

                return out

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

    def lm_logits(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                logits = self.lm
                if logits is not None:
                    if mask is not None:
                        logits *= mask[..., None]

                return logits

    def cell_proposals(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                cell_proposal = self.cell_proposal
                if cell_proposal is not None:
                    if mask is not None:
                        cell_proposal *= mask[..., None]

                return cell_proposal

    def averaged_inputs(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                v = self.v
                if v is not None:
                    if mask is not None:
                        v *= mask[..., None]

                return v

    def averaged_input_logits(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                w = self.w
                if w is not None:
                    if mask is not None:
                        w *= mask[..., None]

                return w

    def segment_lengths(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                u = self.u
                if u is not None:
                    if mask is not None:
                        u *= mask[..., None]

                return u

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
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            activation=None,
            batch_normalization_decay=0.9,
            normalize_weights=False,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        if bias_initializer is None:
            bias_initializer = 'zeros_initializer'
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
        self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.batch_normalization_decay = batch_normalization_decay
        self.normalize_weights = normalize_weights
        self.reuse = reuse
        self.name = name

        self.dense_layer = None
        self.projection = None

        self.initializer = get_initializer(kernel_initializer, self.session)

        self.built = False

    def build(self, inputs):
        if not self.built:
            if self.units is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.units

            with self.session.as_default():
                with self.session.graph.as_default():
                    self.dense_layer = tf.layers.Dense(
                        out_dim,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        _reuse=self.reuse,
                        name=self.name
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
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=self.name
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
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            layers_inner=3,
            activation_inner=None,
            activation=None,
            batch_normalization_decay=0.9,
            project_inputs=False,
            normalize_weights=False,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias

        self.layers_inner = layers_inner
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        if bias_initializer is None:
            bias_initializer = 'zeros_initializer'
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
        self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
        self.activation_inner = get_activation(activation_inner, session=self.session, training=self.training)
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs
        self.normalize_weights = normalize_weights
        self.reuse = reuse
        self.name = name

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
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None

                        l = tf.layers.Dense(
                            out_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            _reuse=self.reuse,
                            name=name
                        )
                        self.dense_layers.append(l)

                    if self.project_inputs:
                        if self.name:
                            name = self.name + '_projection'
                        else:
                            name = None

                        self.projection = tf.layers.Dense(
                            out_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            _reuse=self.reuse,
                            name=name
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
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None,
                            reuse=self.reuse,
                            scope=name
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.dense_layers[-1](F)
                if self.batch_normalization_decay:
                    if self.name:
                        name = self.name + '_i%d' % (self.layers_inner - 1)
                    else:
                        name = None
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=name
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
            reuse=None,
            session=None,
            name=None
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
                self.reuse = reuse
                self.name = name

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
                        use_bias=self.use_bias,
                        name=self.name
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
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=self.name
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
            reuse=None,
            session=None,
            name=None
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
        self.reuse = reuse
        self.name = name

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

                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None

                        l = tf.keras.layers.Conv1D(
                            out_dim,
                            self.kernel_size,
                            padding=self.padding,
                            strides=cur_strides,
                            use_bias=self.use_bias,
                            name=name
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
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None,
                            reuse=self.reuse,
                            scope=name
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.conv_1d_layers[-1](F)

                if self.batch_normalization_decay:
                    if self.name:
                        name = self.name + '_i%d' % (self.layers_inner - 1)
                    else:
                        name = None
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=name
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
            kernel_initializer='glorot_normal_initializer',
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
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.recurrent_activation = get_activation(recurrent_activation, session=self.session, training=self.training)
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.refeed_outputs = refeed_outputs
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
                        recurrent_activation=self.recurrent_activation,
                        name=self.name
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


class RNNResidualLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            layers_inner=3,
            activation=None,
            activation_inner=None,
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs=False,
            return_sequences=True,
            batch_normalization_decay=None,
            project_inputs=False,
            name=None,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.layers_inner = layers_inner
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.activation_inner = get_activation(activation_inner, session=self.session, training=self.training)
        self.recurrent_activation = get_activation(recurrent_activation, session=self.session, training=self.training)
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.refeed_outputs = refeed_outputs
        self.return_sequences = return_sequences
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs
        self.name = name

        self.rnn_layer = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    RNN = tf.keras.layers.LSTM

                    self.rnn_layers = []

                    for i in range(self.layers_inner):
                        if self.units:
                            output_dim = self.units
                        else:
                            output_dim = inputs.shape[-1]

                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None

                        rnn_layer = RNN(
                            output_dim,
                            return_sequences=self.return_sequences,
                            activation=self.activation,
                            recurrent_activation=self.recurrent_activation,
                            name=name
                        )

                        self.rnn_layers.append(rnn_layer)

                    if self.project_inputs:
                        if self.name:
                            name = self.name + '_projection'
                        else:
                            name = None

                        self.projection = tf.layers.Dense(
                            output_dim,
                            name=name
                        )

                    self.built = True

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                F = inputs
                for i in range(self.layers_inner - 1):
                    F = self.rnn_layers[i](F, mask=mask)

                    if self.batch_normalization_decay:
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

                F = self.rnn_layers[-1](F, mask=mask)

                if self.batch_normalization_decay:
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


class RevNetBlock(object):
    def __init__(
            self,
            training=True,
            layers_inner=1,
            use_bias=True,
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            activation=None,
            batch_normalization_decay=None,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.layers_inner = layers_inner
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        if bias_initializer is None:
            bias_initializer = 'zeros_initializer'
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
        self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
        self.activation = get_activation(activation, training=self.training, session=self.session)
        self.batch_normalization_decay = batch_normalization_decay
        self.reuse = reuse
        self.name = name

        self.regularizer_losses = []
        self.W = []
        if self.use_bias:
            self.b = []
        self.F = []
        self.built = False

    def build(self, inputs, weights=None):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    assert int(inputs.shape[-1]) % 2 == 0, 'Final dimension of inputs to RevNetBlock must have even number of units, saw %d.' % int(inputs.shape[-1])
                    k = int(int(inputs.shape[-1]) / 2)
                    if weights is None:
                        p = 1
                    else:
                        p = weights.shape[-1]

                    for l in range(self.layers_inner):
                        if self.name:
                            name_cur = self.name + '_W_%d' % (l + 1)
                        else:
                            name_cur = 'revnet_W_%d' % (l + 1)
                        W = tf.get_variable(
                            name_cur,
                            shape=[p, k, k],
                            initializer=self.kernel_initializer,
                        )
                        self.regularize(W, self.kernel_regularizer)
                        self.W.append(W)
                        
                        if self.use_bias:
                            if self.name:
                                name_cur = self.name + '_b_%d' % (l + 1)
                            else:
                                name_cur = 'revnet_b_%d' % (l + 1)
                            b = tf.get_variable(
                                name_cur,
                                shape=[p, k],
                                initializer=self.kernel_initializer,
                            )
                            self.regularize(b, self.kernel_regularizer)
                            self.b.append(b)

                    self.built = True

    def _apply_inner(self, X, l, weights=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                W = self.W[l]
                if self.use_bias:
                    b = self.b[l]
                if weights is None:
                    out = tf.matmul(X, W[0])
                    if self.use_bias:
                        out += b[0]
                else:
                    W = tf.tensordot(weights, W, 1)
                    retile = False
                    while len(W.shape) <= len(X.shape):
                        retile = True
                        W = tf.expand_dims(W, axis=-3)
                    if retile:
                        tile_shape = tf.concat([[1], tf.shape(X)[1:-1], [1,1]], axis=0)
                        W = tf.tile(W, tile_shape)
                    out = tf.matmul(W, X[..., None])[..., 0]
                    if self.use_bias:
                        b = tf.tensordot(weights, b, 1)
                        while len(b.shape) < len(X.shape):
                            b = tf.expand_dims(b, axis=-2)
                        out += b

                if self.batch_normalization_decay:
                    if self.name:
                        name_cur = self.name + '_bn_%d' % (l + 1)
                    else:
                        name_cur = 'revnet_bn_%d' % (l + 1)
                    out = tf.contrib.layers.batch_norm(
                        out,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=name_cur
                    )
                if self.activation is not None:
                    out = self.activation(out)

                return out

    def forward(self, inputs, weights=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                assert int(inputs.shape[-1]) % 2 == 0, 'Final dimension of inputs to RevNetBlock must have even number of units, saw %d.' % int(inputs.shape[-1])

                if not self.built:
                    self.build(inputs, weights=weights)

                k = int(int(inputs.shape[-1]) / 2)

                left = inputs[..., :k]
                right = inputs[..., k:]

                new_right = right
                for l in range(self.layers_inner):
                    new_right = self._apply_inner(new_right, l, weights=weights)
                new_right = left + new_right
                new_left = right
                out = tf.concat([new_left, new_right], axis=-1)

                return out

    def backward(self, inputs, weights=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                assert int(inputs.shape[-1]) % 2 == 0, 'Final dimension of inputs to RevNetBlock must have even number of units, saw %d.' % int(inputs.shape[-1])

                if not self.built:
                    self.build(inputs, weights=weights)

                k = int(int(inputs.shape[-1]) / 2)

                left = inputs[..., :k]
                right = inputs[..., k:]

                new_left = left
                for l in range(self.layers_inner):
                    new_left = self._apply_inner(new_left, l, weights=weights)
                new_left = right - new_left
                new_right = left
                out = tf.concat([new_left, new_right], axis=-1)

                return out

    def regularize(self, var, regularizer):
        if regularizer is not None:
            with self.session.as_default():
                with self.session.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)

    def get_regularizer_losses(self):
        return self.regularizer_losses[:]


class RevNet(object):
    def __init__(
            self,
            training=True,
            layers=1,
            layers_inner=1,
            use_bias=True,
            kernel_initializer='glorot_normal_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            activation=None,
            batch_normalization_decay=None,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.layers = layers
        self.layers_inner = layers_inner
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation = activation
        self.batch_normalization_decay = batch_normalization_decay
        self.reuse = reuse
        if name is None:
            self.name = 'revnet'
        else:
            self.name = name

        self.blocks = []
        for l in range(self.layers):
            self.blocks.append(
                RevNetBlock(
                    training=self.training,
                    layers_inner=self.layers_inner,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activation=self.activation,
                    batch_normalization_decay=self.batch_normalization_decay,
                    reuse=self.reuse,
                    name=self.name + '_%d' % l
                )
            )
        self.built = False

    def build(self, inputs, weights=None):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    for l in range(self.layers):
                        self.blocks[l].build(inputs, weights=weights)

                    self.built = True

    def forward(self, inputs, weights=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                if not self.built:
                    self.build(inputs, weights=weights)
                out = inputs
                for l in range(self.layers):
                    b = self.blocks[l]
                    out = b.forward(out, weights=weights)
                return out

    def backward(self, inputs, weights=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                if not self.built:
                    self.build(inputs, weights=weights)
                out = inputs
                for l in range(self.layers-1, -1, -1):
                    b = self.blocks[l]
                    out = b.backward(out, weights=weights)
                return out

    def get_regularizer_losses(self):
        out = []
        for b in self.blocks:
            out += b.get_regularizer_losses()
        return out


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
        encoding_batch_normalization_decay=None,
        session=None,
        name=None
):
    session = get_session(session)
    n_layers = len(units_encoder)
    lambdas = []
    kwargs = {'mask': None}

    if encoding_batch_normalization_decay is None:
        encoding_batch_normalization_decay = batch_normalization_decay

    with session.as_default():
        with session.graph.as_default():
            if pre_cnn:
                if name:
                    name_cur = name + '_preCNN'
                else:
                    name_cur = name

                cnn_layer = Conv1DLayer(
                    cnn_kernel_size,
                    training=training,
                    n_filters=n_feats_in,
                    padding='same',
                    activation=tf.nn.elu,
                    batch_normalization_decay=batch_normalization_decay,
                    session=session,
                    name=name_cur
                )
                lambdas.append(make_lambda(cnn_layer, session=session))


            for l in range(n_layers):
                if name:
                    name_cur = name + '_l%d' % l
                else:
                    name_cur = name

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
                    name=name_cur,
                    session=session
                )

                lambdas.append(make_lambda(rnn_layer, session=session, use_kwargs=True))

            if encoding_batch_normalization_decay:
                if name:
                    name_cur = name + '_BN_l%d' % n_layers
                else:
                    name_cur = name

                def bn(x):
                    return tf.contrib.layers.batch_norm(
                        x,
                        decay=encoding_batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=training,
                        updates_collections=None,
                        name=name_cur
                    )

                lambdas.append(make_lambda(bn, session=session))

            out = compose_lambdas(lambdas, **kwargs)

            return out


def cnn_encoder(
        kernel_size,
        units_encoder,
        training=True,
        inner_activation='tanh',
        activation='tanh',
        resnet_n_layers_inner=None,
        batch_normalization_decay=None,
        encoding_batch_normalization_decay=None,
        session=None,
        name=None
):
    session = get_session(session)
    n_layers = len(units_encoder)
    lambdas = []
    kwargs = {'mask': None} # The mask param isn't used. It's just included to permit a unified encoder interface.

    if encoding_batch_normalization_decay is None:
        encoding_batch_normalization_decay = batch_normalization_decay

    with session.as_default():
        with session.graph.as_default():
            for i in range(n_layers - 1):
                if name:
                    name_cur = name + '_l%d' % i
                else:
                    name_cur = name
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
                        session=session,
                        name=name_cur
                    )
                else:
                    cnn_layer = Conv1DLayer(
                        kernel_size,
                        training=training,
                        n_filters=units_encoder[i],
                        padding='causal',
                        activation=inner_activation,
                        batch_normalization_decay=batch_normalization_decay,
                        session=session,
                        name=name_cur
                    )

                def apply_cnn_layer(x, **kwargs):
                    return cnn_layer(x)
                lambdas.append(apply_cnn_layer)

            flattener = tf.layers.Flatten()
            lambdas.append(make_lambda(flattener, session=session))

            if name:
                name_cur = name + '_FC'
            else:
                name_cur = name
            fully_connected_layer = DenseLayer(
                training=training,
                units=units_encoder[-1],
                activation=activation,
                batch_normalization_decay=encoding_batch_normalization_decay,
                session=session,
                name=name_cur
            )
            lambdas.append(make_lambda(fully_connected_layer, session=session))

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
        encoding_batch_normalization_decay=None,
        session=None,
        name=None
):
    session = get_session(session)
    n_layers = len(units_encoder)
    lambdas = []
    kwargs = {'mask': None}  # The mask param isn't used. It's just included to permit a unified encoder interface.

    if encoding_batch_normalization_decay is None:
        encoding_batch_normalization_decay = batch_normalization_decay

    with session.as_default():
        with session.graph.as_default():
            flattener = tf.layers.Flatten()
            lambdas.append(make_lambda(flattener, session=session))

            for i in range(n_layers - 1):
                if name:
                    name_cur = name + '_l%d' % i
                else:
                    name_cur = name

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
                        session=session,
                        name=name_cur
                    )
                else:
                    dense_layer = DenseLayer(
                        training=training,
                        units=n_timesteps * units_encoder[i],
                        activation=inner_activation,
                        batch_normalization_decay=batch_normalization_decay,
                        session=session,
                        name=name_cur
                    )

                lambdas.append(make_lambda(dense_layer, session=session))


            if name:
                name_cur = name + '_l%d' % n_layers
            else:
                name_cur = name

            dense_layer = DenseLayer(
                training=training,
                units=units_encoder[-1],
                activation=activation,
                batch_normalization_decay=encoding_batch_normalization_decay,
                session=session,
                name=name_cur
            )

            lambdas.append(make_lambda(dense_layer, session=session))

            out = compose_lambdas(lambdas, **kwargs)

            return out


def preprocess_decoder_inputs(
        decoder_in,
        n_timesteps,
        units_decoder,
        training=True,
        decoder_hidden_state_expansion_type='tile',
        decoder_temporal_encoding_type='periodic',
        decoder_temporal_encoding_as_mask=False,
        decoder_temporal_encoding_units=32,
        decoder_temporal_encoding_transform=None,
        decoder_inner_activation=None,
        decoder_temporal_encoding_activation=None,
        decoder_batch_normalization_decay=None,
        decoder_conv_kernel_size=3,
        frame_dim=None,
        step=None,
        mask=None,
        n_pretrain_steps=0,
        name=None,
        session=None,
        float_type='float32'
):
    assert step is not None or not n_pretrain_steps, 'step must be provided when n_pretrain_steps is specified.'
    assert len(decoder_in.shape) <= 3 or frame_dim is not None, 'frame_dim must be provided when the input rank is > 3.'
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            out = decoder_in

            if isinstance(float_type, str):
                FLOAT_TF = getattr(tf, float_type)
            else:
                FLOAT_TF = float_type

            # Damp gradients to the encoder for decoder "pretraining", if used
            if n_pretrain_steps > 0:
                def make_decoder_in_rescaled_gradient(decoder):
                    def decoder_in_rescaled_gradient():
                        steps = tf.cast(n_pretrain_steps, dtype=FLOAT_TF)
                        step_tf = tf.cast(step, dtype=FLOAT_TF)
                        scale = 1 - (steps - step_tf) / steps
                        g = decoder * scale
                        out = g + tf.stop_gradient(decoder - g)

                        return out

                    return decoder_in_rescaled_gradient

                out = tf.cond(
                    step >= n_pretrain_steps,
                    lambda: out,
                    make_decoder_in_rescaled_gradient(out)
                )

            # If more than 1 batch dim, flatten out batch dims
            if len(out.shape) > 3:
                flatten_batch = True
                cur_shape = tf.shape(out)
                batch_leading_dims = cur_shape[:-2]
                final_shape = tf.concat([batch_leading_dims, [n_timesteps, frame_dim]], axis=0)
                flattened_batch_shape = tf.concat(
                    [[tf.reduce_prod(batch_leading_dims)], tf.shape(out)[-2:]],
                    axis=0
                )
                if decoder_temporal_encoding_type:
                    if decoder_temporal_encoding_as_mask:
                        units = out.shape[-1]
                    else:
                        units = decoder_temporal_encoding_units
                    final_shape_temporal_encoding = tf.concat([batch_leading_dims, [n_timesteps, units]], axis=0)
                out = tf.reshape(out, flattened_batch_shape)
            else:
                final_shape = None
                final_shape_temporal_encoding = None
                flatten_batch = False

            # Expand out encoder hidden state into time series, either through tiling or reshaped dense transformation
            if decoder_hidden_state_expansion_type.lower() == 'tile':
                tile_dims = [1] * (len(out.shape) + 1)
                tile_dims[-2] = n_timesteps

                out = tf.tile(
                    out[..., None, :],
                    tile_dims
                )
            elif decoder_hidden_state_expansion_type.lower() == 'dense':
                if name:
                    name_cur = name + '_dense_expansion'
                else:
                    name_cur = name
                out = DenseLayer(
                    training=training,
                    units=n_timesteps * units_decoder[0],
                    activation=decoder_inner_activation,
                    batch_normalization_decay=decoder_batch_normalization_decay,
                    session=session,
                    name=name_cur
                )(out)

                decoder_shape = tf.concat([tf.shape(out)[:-1], [n_timesteps, units_decoder[0]]], axis=0)
                out = tf.reshape(out, decoder_shape)
            else:
                raise ValueError(
                    'Unrecognized decoder hidden state expansion type "%s".' % decoder_hidden_state_expansion_type)

            # Mask time series if needed
            if mask is not None:
                out *= mask[..., None]

            # Create a representation of time to supply to decoder
            if decoder_temporal_encoding_type:
                if decoder_temporal_encoding_transform or not decoder_temporal_encoding_as_mask:
                    temporal_encoding_units = decoder_temporal_encoding_units
                else:
                    temporal_encoding_units = out.shape[-1]

                # Create a trainable matrix of weights by timestep
                if decoder_temporal_encoding_type.lower() == 'weights':
                    temporal_encoding = tf.get_variable(
                        'decoder_time_encoding_src_%s' % name,
                        shape=[1, n_timesteps, temporal_encoding_units],
                        initializer=tf.initializers.random_normal,
                    )
                    temporal_encoding = tf.tile(temporal_encoding, [tf.shape(decoder_in)[0], 1, 1])

                # Create a set of periodic functions with trainable phase and frequency
                elif decoder_temporal_encoding_type.lower() == 'periodic':
                    time = tf.range(1., n_timesteps + 1., dtype=FLOAT_TF)[None, ..., None]
                    frequency_logits = tf.linspace(
                        -2.,
                        2.,
                        temporal_encoding_units
                    )
                    # frequency_logits = tf.get_variable(
                    #     'frequency_logits_%s' % name,
                    #     initializer=frequency_logits
                    # )
                    # phase = tf.get_variable(
                    #     'phase_%s' % name,
                    #     initializer=tf.initializers.random_normal,
                    #     shape=[temporal_encoding_units]
                    # )[None, None, ...]
                    # gain = tf.get_variable(
                    #     'gain_%s' % name,
                    #     initializer=tf.glorot_normal_initializer(),
                    #     shape=[temporal_encoding_units]
                    # )[None, None, ...]
                    frequency = tf.exp(frequency_logits)[None, None, ...]
                    # temporal_encoding = tf.sin(time * frequency + phase) * gain
                    temporal_encoding = tf.sin(time * frequency)

                else:
                    raise ValueError(
                        'Unrecognized decoder temporal encoding type "%s".' % decoder_temporal_encoding_type)

                # Transform the temporal encoding
                if decoder_temporal_encoding_transform:
                    if decoder_temporal_encoding_as_mask:
                        units = out.shape[-1]
                    else:
                        units = decoder_temporal_encoding_units

                    if name:
                        name_cur = name + '_temporal_encoding_transform'
                    else:
                        name_cur = name

                    # RNN transform
                    if decoder_temporal_encoding_transform.lower() == 'rnn':
                        temporal_encoding = RNNLayer(
                            training=training,
                            units=units,
                            activation=tf.tanh,
                            recurrent_activation=tf.sigmoid,
                            return_sequences=True,
                            name=name_cur,
                            session=session
                        )(temporal_encoding)

                    # CNN transform (1D)
                    elif decoder_temporal_encoding_transform.lower() == 'cnn':
                        temporal_encoding = Conv1DLayer(
                            decoder_conv_kernel_size,
                            training=training,
                            n_filters=units,
                            padding='same',
                            activation=decoder_inner_activation,
                            batch_normalization_decay=decoder_batch_normalization_decay,
                            name=name_cur,
                            session=session
                        )(temporal_encoding)

                    # Dense transform
                    elif decoder_temporal_encoding_transform.lower() == 'dense':
                        temporal_encoding = DenseLayer(
                            training=training,
                            units=units,
                            activation=decoder_inner_activation,
                            batch_normalization_decay=decoder_batch_normalization_decay,
                            name=name_cur,
                            session=session
                        )(temporal_encoding)

                    else:
                        raise ValueError(
                            'Unrecognized decoder temporal encoding transform "%s".' % decoder_temporal_encoding_transform)

                # Apply activation function
                if decoder_temporal_encoding_activation:
                    activation = get_activation(
                        decoder_temporal_encoding_activation,
                        session=session,
                        training=training
                    )
                    temporal_encoding = activation(temporal_encoding)

                # Apply temporal encoding, either as mask or as extra features
                if decoder_temporal_encoding_as_mask:
                    temporal_encoding = tf.sigmoid(temporal_encoding)
                    out = out * temporal_encoding
                else:
                    temporal_encoding = tf.tile(temporal_encoding, [tf.shape(out)[0], 1, 1])
                    out = tf.concat([out, temporal_encoding], axis=-1)
            else:
                temporal_encoding = None
                final_shape_temporal_encoding = None

            return out, temporal_encoding, flatten_batch, final_shape, final_shape_temporal_encoding
