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


def masked_roll(X, roll_ix, batch_ids, weights=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            batch_ids_rolled = tf.manip.roll(
                batch_ids,
                roll_ix,
                axis=0
            )
            mask = tf.cast(tf.equal(batch_ids_rolled, batch_ids), dtype=X.dtype)

            X = tf.manip.roll(
                X,
                roll_ix,
                axis=0
            )

            if weights is not None:
                weights = tf.manip.roll(
                    weights,
                    roll_ix,
                    axis=0
                )

            return X, weights, mask


def mask_and_lag(X, mask=None, weights=None, n_forward=0, n_backward=0, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if mask == weights == None:
                mask = weights = tf.ones(tf.shape(X)[:-1], dtype=X.dtype)
            elif mask == None:
                mask = tf.cast(weights > 0.5, dtype=X.dtype)
            elif weights == None:
                weights = mask

            # Compute batch element IDs to prevent spillover across batch elements
            batch_ix = tf.range(tf.shape(X)[0])
            tile_ix = [1]
            for i in range(1, len(mask.shape)):
                tile_ix.append(tf.shape(mask)[i])
                batch_ix = batch_ix[..., None]
            batch_ix = tf.tile(batch_ix, tile_ix)
            batch_ids = tf.boolean_mask(batch_ix, mask)

            X = tf.boolean_mask(X, mask)
            weights = tf.boolean_mask(weights, mask)

            tile_bwd = [n_backward, 1]
            X_tile_bwd = tile_bwd + [1]
            roll_bwd = tf.range(0, n_backward)
            X_bwd = tf.tile(X[None, ...], X_tile_bwd)
            batch_ids_bwd = tf.tile(batch_ids[None, ...], tile_bwd)
            weights_bwd = tf.tile(weights[None, ...], tile_bwd)
            X_bwd, weights_bwd, mask_bwd = tf.map_fn(
                lambda x: masked_roll(x[0], x[1], batch_ids=x[2], weights=x[3], session=session),
                [X_bwd, roll_bwd, batch_ids_bwd, weights_bwd],
                dtype=(X.dtype, weights.dtype, X.dtype)
            )
            weights_bwd *= mask_bwd
            weights_bwd = tf.transpose(weights_bwd, [1, 0])
            X_bwd = tf.transpose(X_bwd, [1, 0, 2])

            tile_fwd = [n_forward, 1]
            X_tile_fwd = tile_fwd + [1]
            roll_fwd = tf.range(-1, -n_forward - 1, delta=-1)
            X_fwd = tf.tile(X[None, ...], X_tile_fwd)
            batch_ids_fwd = tf.tile(batch_ids[None, ...], tile_fwd)
            weights_fwd = tf.tile(weights[None, ...], tile_fwd)
            X_fwd, weights_fwd, mask_fwd = tf.map_fn(
                lambda x: masked_roll(x[0], x[1], batch_ids=x[2], weights=x[3], session=session),
                [X_fwd, roll_fwd, batch_ids_fwd, weights_fwd],
                dtype=(X.dtype, weights.dtype, X.dtype)
            )
            weights_fwd *= mask_fwd
            weights_fwd = tf.transpose(weights_fwd, [1, 0])
            X_fwd = tf.transpose(X_fwd, [1, 0, 2])

            return X_bwd, weights_bwd, X_fwd, weights_fwd


def mask_and_lag_old(X, mask, n_forward=0, n_backward=0, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            X_src = tf.boolean_mask(X, mask)
            pad_base = [(0,0) for _ in range(len(X_src.shape)-2)]

            # Compute batch element IDs to prevent spillover across batch elements
            batch_ix = tf.range(tf.shape(X)[0])
            tile_ix = [1]
            for i in range(1, len(mask.shape)):
                tile_ix.append(tf.shape(mask)[i])
                batch_ix = batch_ix[..., None]
            batch_ix = tf.tile(batch_ix, tile_ix)
            batch_ids = tf.boolean_mask(batch_ix, mask)

            _X_bwd = []
            _X_fwd = []
            _X_mask_bwd = []
            _X_mask_fwd = []

            for i in range(n_backward):
                if i == 0:
                    _X_cur = X_src
                    _batch_ids_cur = batch_ids
                else:
                    _pad_left = tf.minimum(i, tf.shape(X_src)[-2])
                    _X_cur = tf.pad(X_src[...,:-i,:], pad_base + [(_pad_left,0), (0,0)])
                    _batch_ids_cur = tf.pad(batch_ids[...,:-i], pad_base + [(_pad_left,0)], constant_values=-1)

                _X_mask_cur = tf.cast(
                    tf.equal(_batch_ids_cur, batch_ids),
                    dtype=mask.dtype
                )

                _X_bwd.append(_X_cur)
                _X_mask_bwd.append(_X_mask_cur)

            for i in range(1, n_forward+1):
                _pad_right = tf.minimum(i, tf.shape(X_src)[-2])
                _X_cur = tf.pad(X_src[...,i:,:], pad_base + [(0,_pad_right), (0,0)])
                _batch_ids_cur = tf.pad(batch_ids[...,i:], pad_base + [(0,_pad_right)], constant_values=-1)

                _X_mask_cur = tf.cast(
                    tf.equal(_batch_ids_cur, batch_ids),
                    dtype=mask.dtype
                )

                _X_fwd.append(_X_cur)
                _X_mask_fwd.append(_X_mask_cur)

            if n_backward:
                _X_bwd = tf.stack(_X_bwd, axis=-2)
                _X_mask_bwd = tf.stack(_X_mask_bwd, axis=-1)
            if n_forward:
                _X_fwd = tf.stack(_X_fwd, axis=-2)
                _X_mask_fwd = tf.stack(_X_mask_fwd, axis=-1)

            return _X_bwd, _X_mask_bwd, _X_fwd, _X_mask_fwd


def mask_and_lag_bugged(X, mask, n_forward=0, n_backward=0, session=None):
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

            for i in range(1, n_backward+1):
                _pad_left = tf.minimum(i, tf.shape(X_src)[-2])
                _X_cur = tf.pad(X_src[...,:-i,:], pad_base + [(_pad_left,0), (0,0)])
                _X_bwd.append(_X_cur)

                _X_mask_cur = tf.cast(time_ix >= i, dtype=mask.dtype)
                _X_mask_cur = tf.boolean_mask(_X_mask_cur, time_mask)
                _X_mask_bwd.append(_X_mask_cur)

            for i in range(1, n_forward+1):
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
            if scale is None and isinstance(init, str) and '_' in init:
                try:
                    init_split = init.split('_')
                    scale = float(init_split[-1])
                    init = '_'.join(init_split[:-1])
                except ValueError:
                    pass

            if scale is None:
                scale = 0.001

            if init is None:
                out = None
            elif isinstance(init, str):
                out = getattr(tf.contrib.layers, init)(scale=scale)
            elif isinstance(init, float):
                out = tf.contrib.layers.l2_regularizer(scale=init)
            else:
                out = init

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
    'w',
    'passthru'
]


def hmlstm_state_size(
        units,
        layers,
        units_below=None,
        units_lm=None,
        units_passthru=None,
        return_c=True,
        return_z_prob=True,
        return_lm=True,
        return_cell_proposal=True,
        return_cae=True
):
    if return_lm:
        assert units_lm is not None, 'units_lm must be provided when using return_lm'
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
    if units_lm is not None:
        if isinstance(units_lm, tuple):
            units_lm = list(units_lm)
        if not isinstance(units_lm, list):
            units_lm = [units_lm] * layers
    for l in range(layers):
        size_cur = []
        size_names_cur = []

        for name in HMLSTM_RETURN_SIGNATURE:
            include = False
            value = None
            if name == 'c' and return_c:
                include = True
                value = units[l]
                if l == 0 and units_passthru:
                    value += units_passthru
            elif name == 'h':
                include = True
                value = units[l]
            elif name in ['z', 'z_prob']:
                if l < layers - 1 and (name == 'z' or return_z_prob):
                    include = True
                    value = 1
            elif name == 'lm':
                if return_lm:
                    include = True
                    value = units_lm[l]
            elif name == 'cell_proposal':
                if return_cell_proposal:
                    include = True
                    value = units[l]
                    if l == 0 and units_passthru:
                        value += units_passthru
            elif name in ['u', 'v', 'w']:
                if return_cae and l < layers - 1:
                    include = True
                    if name == 'u':
                        value = 1
                    else:
                        value = units_below[l]
            elif name == 'passthru':
                if l == 0 and units_passthru:
                    include = True
                    value = units_passthru
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
            prefinal_mode='max',
            resnet_n_layers=1,
            one_hot_inputs=False,
            forget_bias=1.0,
            oracle_boundary=False,
            infer_boundary=True,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            boundary_activation='sigmoid',
            prefinal_activation='tanh',
            boundary_noise_sd=None,
            boundary_discretizer=None,
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            topdown_initializer='glorot_uniform_initializer',
            boundary_initializer='glorot_uniform_initializer',
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
            lm_type='srn',
            lm_use_upper=False,
            lm_order_bwd=0,
            lm_order_fwd=1,
            bottomup_dropout=None,
            recurrent_dropout=None,
            topdown_dropout=None,
            boundary_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            refeed_boundary=False,
            use_bias=True,
            boundary_slope_annealing_rate=None,
            state_slope_annealing_rate=None,
            slope_annealing_max=None,
            min_discretization_prob=None,
            trainable_self_discretization=True,
            state_discretizer=None,
            discretize_state_at_boundary=False,
            discretize_final=False,
            nested_boundaries=False,
            state_noise_sd=None,
            bottomup_noise_sd=None,
            recurrent_noise_sd=None,
            topdown_noise_sd=None,
            sample_at_train=True,
            sample_at_eval=False,
            global_step=None,
            implementation=1,
            batch_normalization_decay=None,
            decoder_embedding=None,
            revnet_n_layers=None,
            revnet_n_layers_inner=1,
            revnet_activation='elu',
            revnet_batch_normalization_decay=None,
            n_passthru_neurons=None,
            l2_normalize_states=False,
            bptt=True,
            reuse=None,
            name=None,
            dtype=None,
            epsilon=1e-8,
            session=None
    ):
        self._session = get_session(session)

        assert not temporal_dropout_plug_lm, 'temporal_dropout_plug_lm is currently broken, do not use.'

        with self._session.as_default():
            with self._session.graph.as_default():
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
                self._prefinal_mode = prefinal_mode
                self._resnet_n_layers = resnet_n_layers

                self._return_cae = return_cae
                self._lm = return_lm_predictions or temporal_dropout_plug_lm
                self._lm_type = lm_type
                self._lm_use_upper = lm_use_upper
                self._return_lm_predictions = return_lm_predictions
                self._lm_order_bwd = lm_order_bwd
                self._lm_order_fwd = lm_order_fwd
                self._temporal_dropout_plug_lm = temporal_dropout_plug_lm
                self._one_hot_inputs = one_hot_inputs

                self._forget_bias = forget_bias

                self._oracle_boundary = oracle_boundary
                self._infer_boundary = infer_boundary

                self._activation = get_activation(activation, session=self._session, training=self._training)
                self._tanh_activation = activation == 'tanh'
                self._inner_activation = get_activation(inner_activation, session=self._session, training=self._training)
                self._tanh_inner_activation = inner_activation == 'tanh'
                self._prefinal_activation =  get_activation(prefinal_activation, session=self._session, training=self._training)
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

                self._weight_normalization = weight_normalization
                self._layer_normalization = layer_normalization
                self._refeed_boundary = refeed_boundary
                self._use_timing_unit = False
                self._use_bias = use_bias
                self._boundary_slope_annealing_rate = boundary_slope_annealing_rate
                self._state_slope_annealing_rate = state_slope_annealing_rate
                self._slope_annealing_max = slope_annealing_max
                self._min_discretization_prob = min_discretization_prob
                self._trainable_self_discretization = trainable_self_discretization
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
                self._discretize_final = discretize_final
                self._nested_boundaries = nested_boundaries
                self._state_noise_sd = state_noise_sd
                self._bottomup_noise_sd = bottomup_noise_sd
                self._recurrent_noise_sd = recurrent_noise_sd
                self._topdown_noise_sd = topdown_noise_sd
                self._sample_at_train = sample_at_train
                self._sample_at_eval = sample_at_eval
                self._global_step = global_step
                self._implementation = implementation

                self._batch_normalization_decay = batch_normalization_decay

                self._decoder_embedding = decoder_embedding

                self._revnet_n_layers = revnet_n_layers
                self._revnet_n_layers_inner = revnet_n_layers_inner
                self._revnet_activation = revnet_activation
                self._revnet_batch_normalization_decay = revnet_batch_normalization_decay

                self._n_passthru_neurons = n_passthru_neurons

                self._l2_normalize_states = l2_normalize_states

                self._bptt = bptt

                self._epsilon = epsilon

                self._regularizer_map = {}
                self._regularization_initialized = False

    def _add_regularization(self, var, regularizer):
        if regularizer is not None:
            with self._session.as_default():
                with self._session.graph.as_default():
                    self._regularizer_map[var] = regularizer

    def initialize_regularization(self):
        assert self.built, "Cannot initialize regularization before calling the HMLSTM layer because the weight matrices haven't been built."

        if not self._regularization_initialized:
            for var in tf.trainable_variables(scope=self.name):
                n1, n2 = var.name.split('/')[-2:]
                if 'bias' in n2:
                    self._add_regularization(var, self._bias_regularizer)
                if 'kernel' in n2:
                    if 'bottomup' in n1 or 'revnet' in n1:
                        self._add_regularization(var, self._bottomup_regularizer)
                    elif 'recurrent' in n1:
                        self._add_regularization(var, self._recurrent_regularizer)
                    elif 'topdown' in n1:
                        self._add_regularization(var, self._topdown_regularizer)
                    elif 'boundary' in n1:
                        self._add_regularization(var, self._boundary_regularizer)
            self._regularization_initialized = True

    def get_regularization(self):
        self.initialize_regularization()
        return self._regularizer_map.copy()

    @property
    def state_size(self):
        units_below = [self._input_dims] + self._num_units[:-1]
        units_lm = []
        for l in range(len(units_below)):
            units_lm.append(units_below[l] * (self._lm_order_bwd + self._lm_order_fwd))

        return hmlstm_state_size(
            self._num_units,
            self._num_layers,
            units_below=units_below,
            units_lm=units_lm,
            units_passthru=self._n_passthru_neurons,
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

    def initialize_kernel(self, l, in_dim, out_dim, kernel_initializer, prefinal_mode=None, name=None):
        with self._session.as_default():
            with self._session.graph.as_default():
                kernel_lambdas = []
                depth = self._kernel_depth
                if prefinal_mode is None:
                    prefinal_mode = self._prefinal_mode

                resnet_kernel_initializer = 'glorot_uniform_initializer'

                if prefinal_mode.lower() == 'max':
                    if out_dim > in_dim:
                        prefinal_dim = out_dim
                        d_trans = 0
                    else:
                        prefinal_dim = in_dim
                        d_trans = depth - 1
                elif prefinal_mode.lower() == 'in':
                    prefinal_dim = in_dim
                    d_trans = depth - 1
                elif prefinal_mode.lower() == 'out':
                    prefinal_dim = out_dim
                    d_trans = 0
                else:
                    raise ValueError('Unrecognized value for prefinal_mode: %s.' % prefinal_mode)

                for d in range(depth):
                    if d == depth - 1:
                        activation = None
                        units = out_dim
                        use_bias = False
                    else:
                        activation = self._prefinal_activation
                        units = prefinal_dim
                        use_bias = self._use_bias
                    if d == d_trans:
                        dense_kernel_initializer = kernel_initializer
                    else:
                        dense_kernel_initializer = 'identity_initializer'
                    if name:
                        name_cur = name + '_l%d_d%d' % (l, d)
                    else:
                        name_cur = 'l%d_d%d' % (l, d)

                    if self._resnet_n_layers > 1:
                        kernel_layer = DenseResidualLayer(
                            training=self._training,
                            units=units,
                            use_bias=use_bias,
                            kernel_initializer=resnet_kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            layers_inner=self._resnet_n_layers,
                            activation_inner=self._prefinal_activation,
                            activation=activation,
                            batch_normalization_decay=self._batch_normalization_decay,
                            project_inputs=False,
                            normalize_weights=self._weight_normalization,
                            reuse=tf.AUTO_REUSE,
                            session=self._session,
                            name=name_cur
                        )
                    else:
                        kernel_layer = DenseLayer(
                            training=self._training,
                            units=units,
                            use_bias=use_bias,
                            kernel_initializer=dense_kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            activation=activation,
                            batch_normalization_decay=self._batch_normalization_decay,
                            normalize_weights=self._weight_normalization,
                            session=self._session,
                            reuse=tf.AUTO_REUSE,
                            name=name_cur
                        )

                    kernel_lambdas.append(make_lambda(kernel_layer, session=self._session))

                kernel = compose_lambdas(kernel_lambdas)

                return kernel

    def build(self, inputs_shape):
        with self._session.as_default():
            with self._session.graph.as_default():
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
                    if self._lm_type.lower() == 'srn':
                        self._kernel_lm_bottomup = []
                        self._kernel_lm_recurrent = []
                    else:
                        self._kernel_lm = []
                    self._lm_output_gain = []
                if self._return_cae:
                    self._kernel_w = []
                if self._revnet_n_layers:
                    self._revnet = []

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
                        bottomup_dim = inputs_shape[1].value - n_boundary_dims
                    else:
                        bottomup_dim = self._num_units[l - 1]
                    bottomup_dim += self._refeed_boundary and (self._implementation == 1)

                    recurrent_dim = self._num_units[l]

                    output_dim = self._num_units[l]
                    if l == 0 and self._n_passthru_neurons:
                        output_dim += self._n_passthru_neurons
                    output_dim *= 4 # forget, input, and output gates, plus cell proposal
                    if self._implementation == 1:
                        if l < self._num_layers - 1:
                            output_dim += 1

                    # Build bias
                    if not self._layer_normalization and self._use_bias:
                        bias = self.add_variable(
                            'bias_%d' % l,
                            shape=[1, output_dim],
                            initializer=self._bias_initializer
                        )
                        self._bias.append(bias)

                    # Build HM-LSTM kernels (bottomup, recurrent, and topdown)
                    self._kernel_bottomup.append(
                        self.initialize_kernel(
                            l,
                            bottomup_dim,
                            output_dim,
                            self._bottomup_initializer,
                            name='bottomup'
                        )
                    )

                    self._kernel_recurrent.append(
                        self.initialize_kernel(
                            l,
                            recurrent_dim,
                            output_dim,
                            self._recurrent_initializer,
                            name='recurrent'
                        )
                    )

                    if l < self._num_layers - 1:
                        topdown_dim = self._num_units[l + 1]

                        self._kernel_topdown.append(
                            self.initialize_kernel(
                                l,
                                topdown_dim,
                                output_dim,
                                self._topdown_initializer,
                                name='topdown'
                            )
                        )

                    # Build boundary (and, if necessary, boundary kernel)
                    if l < self._num_layers - 1 and self._infer_boundary:
                        if self._implementation == 1 and not self._layer_normalization and self._use_bias:
                            bias_boundary = bias[:, -1:]
                            self._bias_boundary.append(bias_boundary)
                        elif self._implementation == 2:
                            # boundary_in_dim = self._num_units[l] * 2
                            boundary_in_dim = self._num_units[l]
                            if l == 0 and self._n_passthru_neurons:
                                boundary_in_dim += self._n_passthru_neurons * 4

                            self._kernel_boundary.append(
                                self.initialize_kernel(
                                    l,
                                    boundary_in_dim,
                                    1,
                                    self._boundary_initializer,
                                    prefinal_mode='in',
                                    name='boundary'
                                )
                            )

                    # Build language model kernel(s)
                    if self._lm:
                        lm_out_dim = bottomup_dim * (self._lm_order_bwd + self._lm_order_fwd)
                        if self._lm_type.lower() == 'srn':
                            lm_out_dim = bottomup_dim * (self._lm_order_bwd + self._lm_order_fwd)
                            if self._lm_use_upper:
                                lm_bottomup_in_dim = sum(self._num_units[l:])
                            else:
                                lm_bottomup_in_dim = self._num_units[l]
                            if l == 0:
                                if self._decoder_embedding is not None:
                                    lm_bottomup_in_dim += int(self._decoder_embedding.shape[-1])
                                if self._n_passthru_neurons:
                                    lm_bottomup_in_dim += self._n_passthru_neurons
                            lm_recurrent_in_dim = bottomup_dim * (self._lm_order_bwd + self._lm_order_fwd)
                            if l == 0 and self._decoder_embedding is not None:
                                lm_recurrent_in_dim += int(self._decoder_embedding.shape[-1])

                            self._lm_output_gain.append(
                                tf.nn.softplus(
                                    self.add_variable(
                                        'lm_output_gain_l%d' % l,
                                        initializer=tf.constant_initializer(np.log(np.e - 1)),
                                        shape=[]
                                    )
                                )
                            )

                            self._kernel_lm_bottomup.append(
                                self.initialize_kernel(
                                    l,
                                    lm_bottomup_in_dim,
                                    lm_out_dim,
                                    self._bottomup_initializer,
                                    name='lm_bottomup'
                                )
                            )

                            self._kernel_lm_recurrent.append(
                                self.initialize_kernel(
                                    l,
                                    lm_recurrent_in_dim,
                                    lm_out_dim,
                                    self._recurrent_initializer,
                                    name='lm_recurrent'
                                )
                            )

                        else:
                            if self._lm_use_upper:
                                lm_in_dim = sum(self._num_units[l:])
                            else:
                                lm_in_dim = self._num_units[l]
                            if l == 0:
                                if self._decoder_embedding is not None:
                                    lm_in_dim += int(self._decoder_embedding.shape[-1])
                                if self._n_passthru_neurons:
                                    lm_in_dim += self._n_passthru_neurons
                            self._kernel_lm.append(
                                self.initialize_kernel(
                                    l,
                                    lm_in_dim,
                                    lm_out_dim,
                                    self._topdown_initializer,
                                    name='lm'
                                )
                            )

                    # Build correspondence AE kernel
                    if self._return_cae:
                        w_out_dim = bottomup_dim
                        if self._lm_use_upper:
                            w_in_dim = sum(self._num_units[l:])
                        else:
                            w_in_dim = self._num_units[l]
                        if l == 0:
                            if self._decoder_embedding is not None:
                                w_in_dim += int(self._decoder_embedding.shape[-1])
                            if self._n_passthru_neurons:
                                w_in_dim += self._n_passthru_neurons

                        self._kernel_w.append(
                            self.initialize_kernel(
                                l,
                                w_in_dim,
                                w_out_dim,
                                self._topdown_initializer,
                                name='w'
                            )
                        )

                    # Build RevNet
                    if self._revnet_n_layers:
                        revnet = RevNet(
                            training=self._training,
                            layers=self._revnet_n_layers,
                            layers_inner=self._revnet_n_layers_inner,
                            kernel_initializer='glorot_uniform_initializer',
                            activation=self._revnet_activation,
                            use_bias=self._use_bias,
                            batch_normalization_decay=self._revnet_batch_normalization_decay,
                            session=self._session,
                            reuse=tf.AUTO_REUSE,
                            name='revnet_l%d' % l
                        )
                        self._revnet.append(revnet)

                # Initialize slope annealing
                if self._infer_boundary:
                    if self._boundary_slope_annealing_rate and self._global_step is not None:
                        rate = self._boundary_slope_annealing_rate
                        if self._slope_annealing_max is None:
                            self.boundary_slope_coef = 1 + rate * tf.cast(self._global_step, dtype=tf.float32)
                        else:
                            self.boundary_slope_coef = tf.minimum(self._slope_annealing_max, 1 + rate * tf.cast(self._global_step, dtype=tf.float32))
                    else:
                        self.boundary_slope_coef = None
                else:
                    self.boundary_slope_coef = None

                if self._state_slope_annealing_rate and self._global_step is not None:
                    rate = self._state_slope_annealing_rate
                    if self._slope_annealing_max is None:
                        self.state_slope_coef = 1 + rate * tf.cast(self._global_step, dtype=tf.float32)
                    else:
                        self.state_slope_coef = tf.minimum(self._slope_annealing_max, 1 + rate * tf.cast(self._global_step, dtype=tf.float32))
                else:
                    self.state_slope_coef = None

        self.built = True

    def call(self, inputs, state):
        with self._session.as_default():
            with self._session.graph.as_default():
                name_maps = []

                if self._oracle_boundary:
                    n_boundary_dims = self._num_layers - 1
                    h_below = inputs[:, :-n_boundary_dims]
                else:
                    h_below = inputs

                z_below = None

                for l, layer in enumerate(state):
                    h_below_clean = h_below
                    if not self._bptt:
                        h_below = tf.stop_gradient(h_below)

                    if self._revnet_n_layers:
                        h_below = self._revnet[l].forward(h_below)

                    if self._bottomup_noise_sd:
                        h_below = tf.cond(
                            self._training,
                            lambda: h_below + tf.random_normal(shape=tf.shape(h_below), stddev=self._bottomup_noise_sd),
                            lambda: h_below
                        )

                    # z_behind: Previous boundary probability at current layer (implicitly 1 if final layer)
                    if l < self._num_layers - 1:
                        z_behind = layer.z
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

                    # Alias useful variables
                    if l < self._num_layers - 1:
                        # Use inner activation if non-final layer
                        activation = self._inner_activation
                    else:
                        # Use outer activation if final layer
                        activation = self._activation
                    units = self._num_units[l]
                    if l == 0 and self._n_passthru_neurons:
                        units += self._n_passthru_neurons

                    # EXTRACT DEPENDENCIES (c_behind, h_behind, h_above, h_below, z_behind, z_below):

                    # c_behind: Previous cell state at current layer
                    c_behind = layer.c
                    if not self._bptt:
                        c_behind = tf.stop_gradient(c_behind)

                    # h_behind: Previous hidden state at current layer
                    h_behind = layer.h
                    if not self._bptt:
                        h_behind = tf.stop_gradient(h_behind)
                    if self._recurrent_noise_sd:
                        h_behind = tf.cond(
                            self._training,
                            lambda: h_behind + tf.random_normal(shape=tf.shape(h_behind), stddev=self._recurrent_noise_sd),
                            lambda: h_behind
                        )
                    h_behind = self._recurrent_dropout(h_behind)

                    # h_above: Hidden state of layer above at previous timestep (implicitly 0 if final layer)
                    if l < self._num_layers - 1:
                        h_above = state[l + 1].h
                        if not self._bptt:
                            h_above = tf.stop_gradient(h_above)
                        if self._topdown_noise_sd:
                            h_above = tf.cond(
                                self._training,
                                lambda: h_above + tf.random_normal(shape=tf.shape(h_above), stddev=self._topdown_noise_sd),
                                lambda: h_above
                            )
                        h_above = self._topdown_dropout(h_above)
                    else:
                        h_above = None

                    if self._implementation == 1 and self._refeed_boundary:
                        h_below = tf.concat([z_behind, h_below], axis=1)

                    if l > 0 and self._state_discretizer and self._discretize_state_at_boundary:
                        h_below_prob = (h_below + 1) / 2
                        h_below_discrete = self._state_discretizer(h_below_prob)

                        if self._min_discretization_prob is None:
                            h_below = h_below_discrete
                        else:
                            def train_func():
                                a = self._min_discretization_prob
                                discretization_prob = 2 * (1 - a) * tf.abs(h_below_prob - 0.5) + a
                                if self._trainable_self_discretization:
                                    discretize = self._state_discretizer(discretization_prob)
                                    discrete = h_discrete * discretize
                                    continuous = h_prob * (1 - discretize)
                                    out = discrete + continuous
                                else:
                                    # if self._sample_at_train:
                                    #     discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                    # else:
                                    #     discretize = discretization_prob < 0.5
                                    discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                    out = tf.where(discretize, h_below_discrete, h_below_prob)
                                return out

                            def eval_func():
                                return h_below_discrete

                            h_below = tf.cond(self._training, train_func, eval_func)

                        h_below = h_below * 2 - 1

                    h_below = self._bottomup_dropout(h_below)

                    s_bottomup = self._kernel_bottomup[l](h_below)
                    if l > 0:
                        s_bottomup *= z_below

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
                        s_topdown = self._kernel_topdown[l](h_above) * z_behind
                        normalizer += z_behind
                        # Add in top-down features
                        s = s + s_topdown

                    s /= normalizer

                    if self._state_noise_sd:
                        s = tf.cond(self._training, lambda: s + tf.random_normal(shape=tf.shape(s), stddev=self._state_noise_sd), lambda: s)

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
                    c = update_prob * c_update + flush_prob * c_flush + copy_prob * c_copy

                    # Compute the gated output of non-copy cell state
                    h = c
                    if self._state_discretizer and self._state_slope_annealing_rate and self._global_step is not None:
                        h *= self.state_slope_coef
                    h = activation(h) * o

                    if l == 0 and self._n_passthru_neurons:
                        passthru = h[:, :self._n_passthru_neurons]
                        h = h[:, self._n_passthru_neurons:]
                    else:
                        passthru = None

                    # if self._state_discretizer and not self._discretize_state_at_boundary and self._global_step is not None:
                    if l < (self._num_layers - 1 + self._discretize_final) \
                            and self._state_discretizer \
                            and not self._discretize_state_at_boundary:
                        # Squash to [-1,1] if state activation is not tanh
                        if (l < self._num_layers - 1 and not self._tanh_inner_activation) \
                                or (l == self._num_layers - 1 and not self._tanh_activation):
                            h_prob = tf.tanh(h)
                        else:
                            h_prob = h
                        h_prob = (h_prob + 1) / 2
                        h_discrete = self._state_discretizer(h_prob)

                        if self._min_discretization_prob is None:
                            h = h_discrete
                        else:
                            def train_func():
                                a = self._min_discretization_prob
                                discretization_prob = 2 * (1 - a) * tf.abs(h_prob - 0.5) + a
                                if self._trainable_self_discretization:
                                    discretize = self._state_discretizer(discretization_prob)
                                    discrete = h_discrete * discretize
                                    continuous = h_prob * (1 - discretize)
                                    out = discrete + continuous
                                else:
                                    # if self._sample_at_train:
                                    #     discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                    # else:
                                    #     discretize = discretization_prob < 0.5
                                    discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                    out = tf.where(discretize, h_discrete, h_prob)
                                return out

                            def eval_func():
                                return h_discrete

                            h = tf.cond(self._training, train_func, eval_func)

                        h = h * 2 - 1

                    elif self._l2_normalize_states:
                        h = tf.nn.l2_normalize(h, axis=-1, epsilon=self._epsilon)

                    # Mix gated output with the previous hidden state proportionally to the copy probability
                    h = copy_prob * h_behind + (1 - copy_prob) * h

                    # Compute the current boundary probability
                    if l < self._num_layers - 1:
                        if self._oracle_boundary:
                            inputs_last_dim = inputs.shape[-1] - self._num_layers + 1
                            z_prob = inputs[:, inputs_last_dim + l:inputs_last_dim + l+1]
                        else:
                            z_prob = tf.zeros([tf.shape(inputs)[0], 1])
                        if self._infer_boundary:
                            z_prob_oracle = z_prob
                            if self._implementation == 2:
                                # In implementation 2, boundary is computed by linear transform of h
                                # z_in = [h, c]
                                z_in = [h]
                                if self._refeed_boundary:
                                    z_in.append(z_behind)

                                if len(z_in) == 1:
                                    z_in = z_in[0]
                                else:
                                    z_in = tf.concat(z_in, axis=1)
                                z_in = self._boundary_dropout(z_in)
                                # In implementation 2, boundary is a function of the hidden state
                                z_logit = self._kernel_boundary[l](z_in)

                            if self._boundary_discretizer and self._boundary_slope_annealing_rate and self._global_step is not None:
                                z_logit *= self.boundary_slope_coef

                            if self.boundary_noise_sd:
                                z_logit = tf.cond(self._training, lambda: z_logit + tf.random_normal(shape=tf.shape(z_logit), stddev=self.boundary_noise_sd), lambda: z_logit)

                            z_prob = self._boundary_activation(z_logit)

                            if self._oracle_boundary:
                                z_prob = tf.maximum(z_prob, z_prob_oracle)

                            if self._nested_boundaries and l > 0 and not self._boundary_discretizer:
                                z_prob *= z_below

                            if self._boundary_discretizer:
                                z_discrete = self._boundary_discretizer(z_prob)

                                if self._min_discretization_prob is None:
                                    z = z_discrete
                                else:
                                    def train_func():
                                        a = self._min_discretization_prob
                                        discretization_prob = 2 * (1 - a) * tf.abs(z_prob - 0.5) + a
                                        if self._trainable_self_discretization:
                                            discretization_decision = self._boundary_discretizer(discretization_prob)
                                            discrete = z_discrete * discretization_decision
                                            continuous = z_prob * (1 - discretization_decision)
                                            out = discrete + continuous
                                        else:
                                            # if self._sample_at_train:
                                            #     discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                            # else:
                                            #     discretize = discretization_prob < 0.5
                                            discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                            out = tf.where(discretize, z_discrete, z_prob)
                                        return out

                                    def eval_func():
                                        return z_discrete

                                    z = tf.cond(self._training, train_func, eval_func)

                            else:
                                z = z_prob

                            if self._nested_boundaries and l > 0 and self._boundary_discretizer:
                                z = z * z_below

                        else:
                            z = z_prob
                        # z = replace_gradient(tf.identity, lambda x: x * 0.1, session=self._session)(z)
                    else:
                        z_prob = None
                        z = None

                    if False:
                        z_comp = 1 - z
                        a_prev = state.a
                        a_cur = h_below_clean
                        a_lt = z_comp * a_prev + z * a_cur
                        a_gt = a_cur
                        a = tf.maximum(a_lt, a_gt)

                    name_map = {
                        'c': c,
                        'h': h,
                        'z': z,
                        'z_prob': z_prob,
                        'cell_proposal': c_flush,
                        'passthru': passthru
                    }

                    h_below = h
                    z_below = z

                    # Clear variables to prevent accidental reuse from wrong layer
                    c = None
                    h = None
                    z = None
                    z_prob = None
                    cell_proposal = None
                    passthru = None

                    name_maps.append(name_map)

                # Compute any required predictions now that entire encoder state is available
                new_state = []
                for l in range(self._num_layers - 1, -1, -1):
                    name_map = name_maps[l]
                    h = name_map['h']
                    z = name_map['z']
                    passthru = name_map['passthru']

                    if self._lm:
                        if self._lm_use_upper:
                            lm_h_in_prob = [s['h'] for s in name_maps[l:]]
                            if len(lm_h_in_prob) > 1:
                                lm_h_in_prob = tf.concat(lm_h_in_prob, axis=-1)
                            else:
                                lm_h_in_prob = lm_h_in_prob[0]
                        else:
                            lm_h_in_prob = h
                        if l < self._num_layers - 1 and self._state_discretizer and self._discretize_state_at_boundary:
                            lm_h_in_prob = (lm_h_in_prob + 1) / 2
                            lm_h_in_discrete = self._state_discretizer(lm_h_in_prob)

                            if self._min_discretization_prob is None:
                                lm_h_in = lm_h_in_discrete
                            else:
                                def train_func():
                                    a = self._min_discretization_prob
                                    discretization_prob = 2 * (1 - a) * tf.abs(lm_h_in_prob - 0.5) + a
                                    if self._trainable_self_discretization:
                                        discretization_decision = self._state_discretizer(discretization_prob)
                                        discrete = lm_h_in_discrete * discretization_decision
                                        continuous = lm_h_in_prob * (1 - discretization_decision)
                                        out = discrete + continuous
                                    else:
                                        # if self._sample_at_train:
                                        #     discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                        # else:
                                        #     discretize = discretization_prob < 0.5
                                        discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                        out = tf.where(discretize, lm_h_in_discrete, lm_h_in_prob)
                                    return out

                                def eval_func():
                                    return lm_h_in_discrete

                                lm_h_in = tf.cond(self._training, train_func, eval_func)

                            lm_h_in = lm_h_in * 2 - 1
                        else:
                            lm_h_in = lm_h_in_prob

                        if self._lm_type.lower() == 'srn':
                            lm_behind = state[l].lm
                            if l == 0:
                                lm_bottomup_in = [lm_h_in]
                                lm_recurrent_in = [lm_behind]
                                if self._decoder_embedding is not None:
                                    lm_bottomup_in.insert(0, self._decoder_embedding)
                                    lm_recurrent_in.insert(0, self._decoder_embedding)
                                if self._n_passthru_neurons:
                                    lm_bottomup_in.insert(0, passthru)
                                    lm_recurrent_in.insert(0, passthru)
                                if len(lm_bottomup_in) > 1:
                                    lm_bottomup_in = tf.concat(lm_bottomup_in, axis=-1)
                                else:
                                    lm_bottomup_in = lm_bottomup_in[0]
                                if len(lm_recurrent_in) > 1:
                                    lm_recurrent_in = tf.concat(lm_recurrent_in, axis=-1)
                                else:
                                    lm_recurrent_in = lm_recurrent_in[0]
                            else:
                                lm_bottomup_in = lm_h_in
                                lm_recurrent_in = lm_behind
                            lm = self._kernel_lm_recurrent[l](lm_recurrent_in) + self._kernel_lm_bottomup[l](lm_bottomup_in)
                            lm = self._inner_activation(lm)
                            lm *= self._lm_output_gain[l]
                        else:
                            if l == 0:
                                lm_in = [lm_h_in]
                                if self._decoder_embedding is not None:
                                    lm_in.insert(0, self._decoder_embedding)
                                if self._n_passthru_neurons:
                                    lm_in.insert(0, passthru)
                                if len(lm_in) > 1:
                                    lm_in = tf.concat(lm_in, axis=-1)
                                else:
                                    lm_in = lm_in[0]
                            else:
                                lm_in = lm_h_in
                            lm = self._kernel_lm[l](lm_in)

                    else:
                        lm = None

                    name_map['lm'] = lm

                    if self._return_cae and l < self._num_layers - 1:
                        v_behind = state[l].v
                        u_behind = state[l].u
                        v = (v_behind * u_behind + h_below_clean) / (u_behind + 1)
                        z_comp = 1 - z
                        u = u_behind * z_comp + z_comp

                        if self._lm_use_upper:
                            w_h_in_prob = [s['h'] for s in name_maps[l:]]
                            if len(lm_h_in_prob) > 1:
                                w_h_in_prob = tf.concat(w_h_in_prob, axis=-1)
                            else:
                                w_h_in_prob = w_h_in_prob[0]
                        else:
                            w_h_in_prob = h
                        if l < self._num_layers - 1 and self._state_discretizer and self._discretize_state_at_boundary:
                            w_h_in_prob = (w_h_in_prob + 1) / 2
                            w_h_in_discrete = self._state_discretizer(w_h_in_prob)

                            if self._min_discretization_prob is None:
                                w_h_in = w_h_in_discrete
                            else:
                                def train_func():
                                    a = self._min_discretization_prob
                                    discretization_prob = 2 * (1 - a) * tf.abs(w_h_in_prob - 0.5) + a
                                    if self._trainable_self_discretization:
                                        discretization_decision = self._state_discretizer(discretization_prob)
                                        discrete = w_h_in_discrete * discretization_decision
                                        continuous = w_h_in_prob * (1 - discretization_decision)
                                        out = discrete + continuous
                                    else:
                                        # if self._sample_at_train:
                                        #     discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                        # else:
                                        #     discretize = discretization_prob < 0.5
                                        discretize = tf.random_uniform(tf.shape(discretization_prob)) < discretization_prob
                                        out = tf.where(discretize, w_h_in_discrete, w_h_in_prob)
                                    return out

                                def eval_func():
                                    return w_h_in_discrete

                                w_h_in = tf.cond(self._training, train_func, eval_func)

                            w_h_in = w_h_in * 2 - 1
                        else:
                            w_h_in = w_h_in_prob

                        w_in = w_h_in
                        if l == 0:
                            w_in = [w_in]
                            if self._decoder_embedding is not None:
                                w_in.insert(0, self._decoder_embedding)
                            if self._n_passthru_neurons:
                                w_in.insert(0, passthru)
                            if len(w_in) > 1:
                                w_in = tf.concat(w_in, axis=-1)
                            else:
                                w_in = w_in[0]
                        w = self._kernel_w[l](w_in)
                    else:
                        v = u = w = None

                    name_map['v'] = v
                    name_map['u'] = u
                    name_map['w'] = w

                    new_state_names = []
                    new_state_cur = []

                    for name in HMLSTM_RETURN_SIGNATURE:
                        include = False
                        if name in ['c', 'h', 'cell_proposal']:
                            include = True
                        elif name in ['z', 'z_prob'] and l < self._num_layers - 1:
                            include = True
                        elif self._lm and name == 'lm':
                            include = True
                        elif self._return_cae and l < self._num_layers - 1 and name in ['u', 'v', 'w']:
                            include = True
                        elif self._n_passthru_neurons and l == 0 and name == 'passthru':
                            include = True

                        if include:
                            new_state_names.append(name)
                            new_state_cur.append(name_map[name])

                    state_tuple = collections.namedtuple('HMLSTMStateTupleL%d' % l, ' '.join(new_state_names))
                    new_state_cur = state_tuple(*new_state_cur)

                    # Append layer to new state
                    new_state.insert(0, new_state_cur)

                return new_state, new_state


class HMLSTMSegmenter(object):
    def __init__(
            self,
            num_units,
            num_layers,
            training=False,
            kernel_depth=2,
            prefinal_mode='max',
            resnet_n_layers=1,
            one_hot_inputs=False,
            forget_bias=1.0,
            oracle_boundary=False,
            infer_boundary=True,
            activation='tanh',
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            boundary_activation='sigmoid',
            prefinal_activation='tanh',
            boundary_discretizer=None,
            boundary_noise_sd=None,
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            topdown_initializer='glorot_uniform_initializer',
            boundary_initializer='glorot_uniform_initializer',
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
            lm_type='srn',
            lm_use_upper=False,
            lm_order_bwd=0,
            lm_order_fwd=1,
            recurrent_dropout=None,
            topdown_dropout=None,
            boundary_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            refeed_boundary=False,
            use_bias=True,
            boundary_slope_annealing_rate=None,
            state_slope_annealing_rate=None,
            slope_annealing_max=None,
            min_discretization_prob=None,
            trainable_self_discretization=True,
            state_discretizer=None,
            discretize_state_at_boundary=None,
            discretize_final=False,
            nested_boundaries=False,
            state_noise_sd=None,
            bottomup_noise_sd=None,
            recurrent_noise_sd=None,
            topdown_noise_sd=None,
            sample_at_train=True,
            sample_at_eval=False,
            global_step=None,
            implementation=1,
            decoder_embedding=None,
            revnet_n_layers=None,
            revnet_n_layers_inner=1,
            revnet_activation='elu',
            revnet_batch_normalization_decay=None,
            n_passthru_neurons=None,
            l2_normalize_states=False,
            bptt=True,
            reuse=None,
            name=None,
            dtype=None,
            epsilon=1e-8,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                if not isinstance(num_units, list):
                    self.num_units = [num_units] * num_layers
                else:
                    self.num_units = num_units

                assert len(self.num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                self.num_layers = num_layers
                self.training = training
                self.kernel_depth = kernel_depth
                self.prefinal_mode = prefinal_mode
                self.resnet_n_layers = resnet_n_layers
                self.one_hot_inputs = one_hot_inputs
                self.forget_bias = forget_bias
                self.oracle_boundary = oracle_boundary
                self.infer_boundary = infer_boundary

                self.activation = activation
                self.inner_activation = inner_activation
                self.recurrent_activation = recurrent_activation
                self.boundary_activation = boundary_activation
                self.prefinal_activation = prefinal_activation
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
                self.lm_type = lm_type
                self.lm_use_upper = lm_use_upper
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
                self.use_bias = use_bias
                self.boundary_slope_annealing_rate = boundary_slope_annealing_rate
                self.state_slope_annealing_rate = state_slope_annealing_rate
                self.slope_annealing_max = slope_annealing_max
                self.min_discretization_prob = min_discretization_prob
                self.trainable_self_discretization = trainable_self_discretization
                self.state_discretizer = state_discretizer
                self.discretize_state_at_boundary = discretize_state_at_boundary
                self.discretize_final = discretize_final
                self.nested_boundaries = nested_boundaries
                self.state_noise_sd = state_noise_sd
                self.bottomup_noise_sd = bottomup_noise_sd
                self.recurrent_noise_sd = recurrent_noise_sd
                self.topdown_noise_sd = topdown_noise_sd
                self.sample_at_train = sample_at_train
                self.sample_at_eval = sample_at_eval
                self.global_step = global_step
                self.implementation = implementation
                self.decoder_embedding = decoder_embedding
                self.revnet_n_layers = revnet_n_layers
                self.revnet_n_layers_inner = revnet_n_layers_inner
                self.revnet_activation = revnet_activation
                self.revnet_batch_normalization_decay = revnet_batch_normalization_decay
                self.n_passthru_neurons = n_passthru_neurons
                self.l2_normalize_states = l2_normalize_states
                self.bptt = bptt
                self.epsilon = epsilon

                self.reuse = reuse
                self.name = name
                self.dtype = dtype
                self.boundary_slope_coef = None
                self.state_slope_coef = None

                self.built = False

    def get_regularizer_losses(self):
        if self.built:
            return self.cell.get_regularization()
        raise ValueError(
            'Attempted to get regularizer losses from an HMLSTMSegmenter that has not been called on data.')

    def build(self, inputs=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.cell = HMLSTMCell(
                    self.num_units,
                    self.num_layers,
                    training=self.training,
                    kernel_depth=self.kernel_depth,
                    prefinal_mode=self.prefinal_mode,
                    resnet_n_layers=self.resnet_n_layers,
                    one_hot_inputs=self.one_hot_inputs,
                    forget_bias=self.forget_bias,
                    oracle_boundary=self.oracle_boundary,
                    infer_boundary=self.infer_boundary,
                    activation=self.activation,
                    inner_activation=self.inner_activation,
                    recurrent_activation=self.recurrent_activation,
                    boundary_activation=self.boundary_activation,
                    prefinal_activation=self.prefinal_activation,
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
                    lm_type=self.lm_type,
                    lm_use_upper=self.lm_use_upper,
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
                    use_bias=self.use_bias,
                    boundary_slope_annealing_rate=self.boundary_slope_annealing_rate,
                    state_slope_annealing_rate=self.state_slope_annealing_rate,
                    slope_annealing_max=self.slope_annealing_max,
                    min_discretization_prob=self.min_discretization_prob,
                    trainable_self_discretization=self.trainable_self_discretization,
                    state_discretizer=self.state_discretizer,
                    discretize_state_at_boundary=self.discretize_state_at_boundary,
                    discretize_final=self.discretize_final,
                    nested_boundaries=self.nested_boundaries,
                    state_noise_sd=self.state_noise_sd,
                    bottomup_noise_sd=self.bottomup_noise_sd,
                    recurrent_noise_sd=self.recurrent_noise_sd,
                    topdown_noise_sd=self.topdown_noise_sd,
                    sample_at_train=self.sample_at_train,
                    sample_at_eval=self.sample_at_eval,
                    global_step=self.global_step,
                    implementation=self.implementation,
                    decoder_embedding=self.decoder_embedding,
                    revnet_n_layers = self.revnet_n_layers,
                    revnet_n_layers_inner = self.revnet_n_layers_inner,
                    revnet_activation = self.revnet_activation,
                    revnet_batch_normalization_decay = self.revnet_batch_normalization_decay,
                    n_passthru_neurons = self.n_passthru_neurons,
                    l2_normalize_states=self.l2_normalize_states,
                    bptt=self.bptt,
                    reuse=self.reuse,
                    name=self.name,
                    dtype=self.dtype,
                    epsilon=self.epsilon,
                    session=self.session
                )

                self.cell.build(inputs.shape[1:])
                if self.revnet_n_layers:
                    self.revnet = self.cell._revnet
                else:
                    self.revnet = None

        self.built = True

    def __call__(self, inputs, mask=None, boundaries=None):
        assert not (self.oracle_boundary and boundaries is None), 'A boundaries arg must be provided when oracle_boundary is true'

        with self.session.as_default():
            with self.session.graph.as_default():
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

    def get_regularization(self):
        return self.cell.get_regularization()


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

    def passthru_neurons(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():

                return self.l[0].passthru_neurons(mask=mask)

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

    def passthru_neurons(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                passthru = self.passthru
                if passthru is not None:
                    if mask is not None:
                        passthru *= mask[..., None]

                return passthru

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



class MaskedLSTMCell(LayerRNNCell):
    def __init__(
            self,
            units,
            training=False,
            kernel_depth=1,
            resnet_n_layers=1,
            prefinal_mode='max',
            forget_bias=1.0,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            bottomup_dropout=None,
            recurrent_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            batch_normalization_decay=None,
            reuse=None,
            name=None,
            dtype=None,
            epsilon=1e-8,
            session=None
    ):
        self._session = get_session(session)
        with self._session.as_default():
            with self._session.graph.as_default():
                super(MaskedLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                self._num_units = units
                self._training = training

                self._kernel_depth = kernel_depth
                self._resnet_n_layers = resnet_n_layers
                self._prefinal_mode = prefinal_mode
                self._forget_bias = forget_bias

                self._activation = get_activation(activation, session=self._session, training=self._training)
                self._prefinal_activation = get_activation(prefinal_activation, session=self._session, training=self._training)
                self._recurrent_activation = get_activation(recurrent_activation, session=self._session, training=self._training)

                self._bottomup_initializer = get_initializer(bottomup_initializer, session=self._session)
                self._recurrent_initializer = get_initializer(recurrent_initializer, session=self._session)
                self._bias_initializer = get_initializer(bias_initializer, session=self._session)

                self._bottomup_regularizer = get_regularizer(bottomup_regularizer, session=self._session)
                self._recurrent_regularizer = get_regularizer(recurrent_regularizer, session=self._session)
                self._bias_regularizer = get_regularizer(bias_regularizer, session=self._session)

                self._bottomup_dropout = get_dropout(bottomup_dropout, training=self._training, session=self._session)
                self._recurrent_dropout = get_dropout(recurrent_dropout, training=self._training, session=self._session)

                self._weight_normalization = weight_normalization
                self._layer_normalization = layer_normalization
                self._use_bias = use_bias
                self._global_step = global_step

                self._batch_normalization_decay = batch_normalization_decay

                self._epsilon = epsilon

                self._regularizer_map = {}
                self._regularization_initialized = False

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(c=self._num_units,h=self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _add_regularization(self, var, regularizer):
        if regularizer is not None:
            with self._session.as_default():
                with self._session.graph.as_default():
                    self._regularizer_map[var] = regularizer

    def initialize_regularization(self):
        assert self.built, "Cannot initialize regularization before calling the LSTM layer because the weight matrices haven't been built."

        if not self._regularization_initialized:
            for var in tf.trainable_variables(scope=self.name):
                n1, n2 = var.name.split('/')[-2:]
                if 'bias' in n2:
                    self._add_regularization(var, self._bias_regularizer)
                if 'kernel' in n2:
                    if 'bottomup' in n1 or 'revnet' in n1:
                        self._add_regularization(var, self._bottomup_regularizer)
                    elif 'recurrent' in n1:
                        self._add_regularization(var, self._recurrent_regularizer)
                    elif 'topdown' in n1:
                        self._add_regularization(var, self._topdown_regularizer)
                    elif 'boundary' in n1:
                        self._add_regularization(var, self._boundary_regularizer)
            self._regularization_initialized = True

    def get_regularization(self):
        self.initialize_regularization()
        return self._regularizer_map.copy()

    def initialize_kernel(self, in_dim, out_dim, kernel_initializer, prefinal_mode=None, name=None):
        with self._session.as_default():
            with self._session.graph.as_default():
                kernel_lambdas = []
                depth = self._kernel_depth
                if prefinal_mode is None:
                    prefinal_mode = self._prefinal_mode

                resnet_kernel_initializer = 'glorot_uniform_initializer'

                if prefinal_mode.lower() == 'max':
                    if out_dim > in_dim:
                        prefinal_dim = out_dim
                        d_trans = 0
                    else:
                        prefinal_dim = in_dim
                        d_trans = depth - 1
                elif prefinal_mode.lower() == 'in':
                    prefinal_dim = in_dim
                    d_trans = depth - 1
                elif prefinal_mode.lower() == 'out':
                    prefinal_dim = out_dim
                    d_trans = 0
                else:
                    raise ValueError('Unrecognized value for prefinal_mode: %s.' % prefinal_mode)

                for d in range(depth):
                    if d == depth - 1:
                        activation = None
                        units = out_dim
                        use_bias = False
                    else:
                        activation = self._prefinal_activation
                        units = prefinal_dim
                        use_bias = self._use_bias
                    if d == d_trans:
                        dense_kernel_initializer = kernel_initializer
                    else:
                        dense_kernel_initializer = 'identity_initializer'
                    if name:
                        name_cur = name + '_d%d' % d
                    else:
                        name_cur = 'd%d' % d

                    if self._resnet_n_layers > 1:
                        kernel_layer = DenseResidualLayer(
                            training=self._training,
                            units=units,
                            use_bias=use_bias,
                            kernel_initializer=resnet_kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            layers_inner=self._resnet_n_layers,
                            activation_inner=self._prefinal_activation,
                            activation=activation,
                            batch_normalization_decay=self._batch_normalization_decay,
                            project_inputs=False,
                            normalize_weights=self._weight_normalization,
                            reuse=tf.AUTO_REUSE,
                            session=self._session,
                            name=name_cur
                        )
                    else:
                        kernel_layer = DenseLayer(
                            training=self._training,
                            units=units,
                            use_bias=use_bias,
                            kernel_initializer=dense_kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            activation=activation,
                            batch_normalization_decay=self._batch_normalization_decay,
                            normalize_weights=self._weight_normalization,
                            session=self._session,
                            reuse=tf.AUTO_REUSE,
                            name=name_cur
                        )

                    kernel_lambdas.append(make_lambda(kernel_layer, session=self._session))

                kernel = compose_lambdas(kernel_lambdas)

                return kernel

    def build(self, inputs_shape):
        with self._session.as_default():
            with self._session.graph.as_default():
                if isinstance(inputs_shape, list): # Has a mask
                    inputs_shape = inputs_shape[0]
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

                self._input_dims = inputs_shape[1].value
                bottomup_dim = self._input_dims
                recurrent_dim = self._num_units
                output_dim = self._num_units * 4 # forget, input, and output gates, plus cell proposal

                # Build bias
                if not self._layer_normalization and self._use_bias:
                    self._bias = self.add_variable(
                        'bias',
                        shape=[1, output_dim],
                        initializer=self._bias_initializer
                    )

                # Build LSTM kernels (bottomup and recurrent)
                self._kernel_bottomup = self.initialize_kernel(
                    bottomup_dim,
                    output_dim,
                    self._bottomup_initializer,
                    name='bottomup'
                )

                self._kernel_recurrent = self.initialize_kernel(
                    recurrent_dim,
                    output_dim,
                    self._recurrent_initializer,
                    name='recurrent'
                )

        self.built = True

    def call(self, inputs, state):
        with self._session.as_default():
            with self._session.graph.as_default():
                if isinstance(inputs, list):
                    inputs, mask = inputs
                else:
                    mask = None

                units = self._num_units
                c_prev = state.c
                h_prev = state.h

                s_bottomup = self._kernel_bottomup(inputs)
                s_recurrent = self._kernel_recurrent(h_prev)
                s = s_bottomup + s_recurrent
                if not self._layer_normalization and self._use_bias:
                    s += self._bias

                # Forget gate
                f = s[:, :units]
                if self._layer_normalization:
                    f = self.norm(f, 'f_ln')
                f = self._recurrent_activation(f + self._forget_bias)

                # Input gate
                i = s[:, units:units * 2]
                if self._layer_normalization:
                    i = self.norm(i, 'i_ln')
                i = self._recurrent_activation(i)

                # Output gate
                o = s[:, units * 2:units * 3]
                if self._layer_normalization:
                    o = self.norm(o, 'o_ln')
                o = self._recurrent_activation(o)

                # Cell proposal
                g = s[:, units * 3:units * 4]
                if self._layer_normalization:
                    g = self.norm(g, 'g_ln')
                g = self._activation(g)

                c = f * c_prev + i * g
                h = o * self._activation(c)

                if mask is not None:
                    c = c * mask + c_prev * (1 - mask)
                    h = h * mask + h_prev * (1 - mask)

                return h, tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)


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
            kernel_initializer='glorot_uniform_initializer',
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
            kernel_initializer='glorot_uniform_initializer',
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
            kernel_initializer='glorot_uniform_initializer',
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
            kernel_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
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
        self.recurrent_initializer = get_initializer(recurrent_initializer, session=self.session)
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
                        kernel_initializer=self.kernel_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        bias_initializer=self.bias_initializer,
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
            kernel_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
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
        self.recurrent_initializer = get_initializer(recurrent_initializer, session=self.session)
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
                            kernel_initializer=self.kernel_initializer,
                            recurrent_initializer=self.recurrent_initializer,
                            bias_initializer=self.bias_initializer,
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
            kernel_initializer='glorot_uniform_initializer',
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


class MaskedLSTMLayer(object):
    def __init__(
            self,
            units=None,
            training=False,
            kernel_depth=1,
            resnet_n_layers=1,
            prefinal_mode='max',
            forget_bias=1.0,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            bottomup_dropout=None,
            recurrent_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            batch_normalization_decay=None,
            return_sequences=True,
            reuse=None,
            name=None,
            dtype=None,
            epsilon=1e-8,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.kernel_depth = kernel_depth
        self.resnet_n_layers = resnet_n_layers
        self.prefinal_mode = prefinal_mode
        self.forget_bias = forget_bias
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.prefinal_activation = prefinal_activation
        self.bottomup_initializer = bottomup_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.bottomup_regularizer = bottomup_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.bottomup_dropout = bottomup_dropout
        self.recurrent_dropout = recurrent_dropout
        self.weight_normalization = weight_normalization
        self.layer_normalization = layer_normalization
        self.use_bias = use_bias
        self.global_step = global_step
        self.batch_normalization_decay = batch_normalization_decay
        self.return_sequences = return_sequences
        self.reuse = reuse
        self.name = name
        self.dtype = dtype
        self.epsilon = epsilon

        self.cell = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():

                    if self.units is None:
                        units = inputs.shape[-1]
                    else:
                        units = self.units
                        
                    self.cell = MaskedLSTMCell(
                        units,
                        training=self.training,
                        kernel_depth=self.kernel_depth,
                        resnet_n_layers=self.resnet_n_layers,
                        prefinal_mode=self.prefinal_mode,
                        forget_bias=self.forget_bias,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        prefinal_activation=self.prefinal_activation,
                        bottomup_initializer=self.bottomup_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        bias_initializer=self.bias_initializer,
                        bottomup_regularizer=self.bottomup_regularizer,
                        recurrent_regularizer=self.recurrent_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        bottomup_dropout=self.bottomup_dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        weight_normalization=self.weight_normalization,
                        layer_normalization=self.layer_normalization,
                        use_bias=self.use_bias,
                        global_step=self.global_step,
                        batch_normalization_decay=self.batch_normalization_decay,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        epsilon=self.epsilon,
                    )

                    self.cell.build(inputs.shape[1:])

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                if mask is None:
                    sequence_length = None
                else:
                    sequence_length = tf.reduce_sum(mask, axis=1)
                    while len(mask.shape) < 3:
                        mask = mask[..., None]
                    inputs = [inputs, mask]

                H, _ = tf.nn.dynamic_rnn(
                    self.cell,
                    inputs,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

                if not self.return_sequences:
                    H = H[:, -1]

                return H

class RevNetBlock(object):
    def __init__(
            self,
            training=True,
            layers_inner=1,
            use_bias=True,
            kernel_initializer='glorot_uniform_initializer',
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
                    with tf.variable_scope(self.name, reuse=self.reuse):
                        assert int(inputs.shape[-1]) % 2 == 0, 'Final dimension of inputs to RevNetBlock must have even number of units, saw %d.' % int(inputs.shape[-1])
                        k = int(int(inputs.shape[-1]) / 2)
                        if weights is None:
                            p = 1
                            shape = [k,k]
                        else:
                            p = weights.shape[-1]
                            shape = [p, k, k]

                        for l in range(self.layers_inner):
                            name_cur = 'kernel_l%d' % l
                            W = tf.get_variable(
                                name_cur,
                                shape=shape,
                                initializer=self.kernel_initializer,
                            )
                            if weights is None:
                                W = W[None, ...]
                            self.regularize(W, self.kernel_regularizer)
                            self.W.append(W)

                            if self.use_bias:
                                name_cur = 'bias_l%d' % l
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
            kernel_initializer='glorot_uniform_initializer',
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
                    name=self.name + '_d%d' % l
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
        decoder_positional_encoding_type='periodic',
        decoder_positional_encoding_as_mask=False,
        decoder_positional_encoding_units=32,
        decoder_positional_encoding_transform=None,
        decoder_positional_encoding_activation=None,
        decoder_inner_activation=None,
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

            FLOAT_TF = getattr(tf, float_type)
            FLOAT_NP = getattr(np, float_type)

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
                if decoder_positional_encoding_type:
                    if decoder_positional_encoding_as_mask:
                        units = out.shape[-1]
                    else:
                        units = decoder_positional_encoding_units
                    final_shape_positional_encoding = tf.concat([batch_leading_dims, [n_timesteps, units]], axis=0)
                out = tf.reshape(out, flattened_batch_shape)
            else:
                final_shape = None
                final_shape_positional_encoding = None
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
            if decoder_positional_encoding_type:
                if decoder_positional_encoding_transform or not decoder_positional_encoding_as_mask:
                    positional_encoding_units = decoder_positional_encoding_units
                else:
                    positional_encoding_units = out.shape[-1]

                # Create a trainable matrix of weights by timestep
                if decoder_positional_encoding_type.lower() == 'weights':
                    positional_encoding = tf.get_variable(
                        'decoder_positional_encoding_src_%s' % name,
                        shape=[1, n_timesteps, positional_encoding_units],
                        initializer=tf.initializers.random_normal,
                    )
                    positional_encoding = tf.tile(positional_encoding, [tf.shape(decoder_in)[0], 1, 1])

                # Create a set of periodic functions with trainable phase and frequency
                elif decoder_positional_encoding_type.lower() in ['periodic', 'transformer_pe']:
                    time = np.arange(1., n_timesteps + 1.)[None, ..., None]
                    n = positional_encoding_units // 2

                    if decoder_positional_encoding_type.lower() == 'periodic':
                        coef = np.exp(np.linspace(-2,2, n))[None, None, ...]
                    elif decoder_positional_encoding_type.lower() == 'transformer_pe':
                        log_timescale_increment = np.log(10000) / (n - 1)
                        coef = (np.exp(np.arange(n) * -log_timescale_increment))[None, None, ...]

                    sin = np.sin(time * coef)
                    cos = np.cos(time * coef)

                    positional_encoding = np.zeros([1, n_timesteps, positional_encoding_units], dtype=FLOAT_NP)
                    positional_encoding[..., 0::2] = sin
                    positional_encoding[..., 1::2] = cos

                else:
                    raise ValueError(
                        'Unrecognized decoder temporal encoding type "%s".' % decoder_positional_encoding_type)

                # Transform the temporal encoding
                if decoder_positional_encoding_transform:
                    if decoder_positional_encoding_as_mask:
                        units = out.shape[-1]
                    else:
                        units = decoder_positional_encoding_units

                    if name:
                        name_cur = name + '_temporal_encoding_transform'
                    else:
                        name_cur = name

                    # RNN transform
                    if decoder_positional_encoding_transform.lower() == 'rnn':
                        positional_encoding = RNNLayer(
                            training=training,
                            units=units,
                            activation=tf.tanh,
                            recurrent_activation=tf.sigmoid,
                            return_sequences=True,
                            name=name_cur,
                            session=session
                        )(positional_encoding)

                    # CNN transform (1D)
                    elif decoder_positional_encoding_transform.lower() == 'cnn':
                        positional_encoding = Conv1DLayer(
                            decoder_conv_kernel_size,
                            training=training,
                            n_filters=units,
                            padding='same',
                            activation=decoder_inner_activation,
                            batch_normalization_decay=decoder_batch_normalization_decay,
                            name=name_cur,
                            session=session
                        )(positional_encoding)

                    # Dense transform
                    elif decoder_positional_encoding_transform.lower() == 'dense':
                        positional_encoding = DenseLayer(
                            training=training,
                            units=units,
                            activation=decoder_inner_activation,
                            batch_normalization_decay=decoder_batch_normalization_decay,
                            name=name_cur,
                            session=session
                        )(positional_encoding)

                    else:
                        raise ValueError(
                            'Unrecognized decoder temporal encoding transform "%s".' % decoder_positional_encoding_transform)

                # Apply activation function
                if decoder_positional_encoding_activation:
                    activation = get_activation(
                        decoder_positional_encoding_activation,
                        session=session,
                        training=training
                    )
                    positional_encoding = activation(positional_encoding)

                # Apply temporal encoding, either as mask or as extra features
                if decoder_positional_encoding_as_mask:
                    positional_encoding = tf.sigmoid(positional_encoding)
                    out = out * positional_encoding
                else:
                    positional_encoding = tf.tile(positional_encoding, [tf.shape(out)[0], 1, 1])
                    out = tf.concat([out, positional_encoding], axis=-1)
            else:
                positional_encoding = None
                final_shape_positional_encoding = None

            return out, positional_encoding, flatten_batch, final_shape, final_shape_positional_encoding
