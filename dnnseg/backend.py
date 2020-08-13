import math
import re
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, rnn_cell_impl, state_ops
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
            # Shift batch indices
            batch_ids_rolled = tf.manip.roll(
                batch_ids,
                roll_ix,
                axis=0
            )

            # Force batch indices to -1 when they loop all they way back, so that they never match the current batch index
            seq_len = tf.shape(batch_ids)[0]
            seq_mask = tf.cond(
                roll_ix < 0,
                lambda: tf.sequence_mask(lengths=seq_len + roll_ix, maxlen=seq_len),
                lambda: tf.reverse(tf.sequence_mask(lengths=seq_len - roll_ix, maxlen=seq_len), axis=[0]),
            )
            batch_ids_rolled = tf.where(
                seq_mask,
                batch_ids_rolled,
                tf.fill(tf.shape(batch_ids_rolled), -1)
            )

            # Keep only timepoints within the same batch item
            mask = tf.cast(tf.equal(batch_ids_rolled, batch_ids), dtype=X.dtype)

            # Shift and mask features
            X = tf.manip.roll(
                X,
                roll_ix,
                axis=0
            ) * mask[..., None]

            if weights is not None:
                # Shift and mask weights
                weights = tf.manip.roll(
                    weights,
                    roll_ix,
                    axis=0
                ) * mask

            return X, weights, mask


def mask_and_lag(X, mask=None, weights=None, n_forward=0, n_backward=0, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            n_forward = max(n_forward, 0)
            n_backward = max(n_backward, 0)

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
            batch_ids_at_pred = batch_ids

            item_shape = tf.shape(mask)[1:]
            time_ix = tf.reshape(tf.range(tf.reduce_prod(item_shape)), item_shape)[None, ...]
            tile_ix = [tf.shape(mask)[0]] + [1] * (len(mask.shape) - 1)
            time_ix = tf.tile(time_ix, tile_ix)
            time_ix = tf.cast(time_ix, dtype=X.dtype)
            time_ids = tf.boolean_mask(time_ix, mask)
            time_ids_at_pred = time_ids

            X = tf.boolean_mask(X, mask)
            weights = tf.boolean_mask(weights, mask)
            
            X = tf.concat([X, time_ids[..., None]], axis=-1)

            # BACKWARD
            # Tile
            n_backward_cur = max(n_backward, 1)
            tile_bwd = [n_backward_cur, 1]
            X_tile_bwd = tile_bwd + [1]
            roll_bwd = tf.range(0, n_backward_cur)
            X_bwd = tf.tile(X[None, ...], X_tile_bwd)
            batch_ids_bwd = tf.tile(batch_ids[None, ...], tile_bwd)
            weights_bwd = tf.tile(weights[None, ...], tile_bwd)

            # Shift
            X_bwd, weights_bwd, mask_bwd = tf.map_fn(
                lambda x: masked_roll(x[0], x[1], batch_ids=x[2], weights=x[3], session=session),
                [X_bwd, roll_bwd, batch_ids_bwd, weights_bwd],
                dtype=(X.dtype, weights.dtype, X.dtype)
            )

            # Transpose
            X_bwd = tf.transpose(X_bwd, [1, 0, 2])
            weights_bwd = tf.transpose(weights_bwd, [1, 0])
            mask_bwd = tf.transpose(mask_bwd, [1, 0])

            # Split feats and time indices
            time_ids_bwd = tf.cast(X_bwd[..., -1], dtype=tf.int32)
            X_bwd = X_bwd[..., :-1]

            # Relativize timesteps to current timestep
            time_ids_0 = time_ids_bwd[..., :1]
            time_ids_bwd = (time_ids_0 - time_ids_bwd) * tf.cast(mask_bwd, dtype=time_ids_0.dtype)
            X_bwd = X_bwd[:,:n_backward]

            # FORWARD
            # Tile
            tile_fwd = [n_forward, 1]
            X_tile_fwd = tile_fwd + [1]
            roll_fwd = tf.range(-1, -n_forward - 1, delta=-1)
            X_fwd = tf.tile(X[None, ...], X_tile_fwd)
            batch_ids_fwd = tf.tile(batch_ids[None, ...], tile_fwd)
            weights_fwd = tf.tile(weights[None, ...], tile_fwd)

            # Shift
            X_fwd, weights_fwd, mask_fwd = tf.map_fn(
                lambda x: masked_roll(x[0], x[1], batch_ids=x[2], weights=x[3], session=session),
                [X_fwd, roll_fwd, batch_ids_fwd, weights_fwd],
                dtype=(X.dtype, weights.dtype, X.dtype)
            )

            # Transpose
            X_fwd = tf.transpose(X_fwd, [1, 0, 2])
            weights_fwd = tf.transpose(weights_fwd, [1, 0])
            mask_fwd = tf.transpose(mask_fwd, [1, 0])

            # Split feats and time indices
            time_ids_fwd = tf.cast(X_fwd[..., -1], dtype=tf.int32)
            X_fwd = X_fwd[..., :-1]

            # Relativize timesteps to current timestep
            time_ids_fwd = (time_ids_fwd - time_ids_0 - 1) * tf.cast(mask_fwd, dtype=time_ids_0.dtype)

            out = {
                'X_bwd': X_bwd,
                'weights_bwd': weights_bwd,
                'time_ids_bwd': time_ids_bwd,
                'batch_ids_bwd': tf.matrix_transpose(batch_ids_bwd),
                'mask_bwd': mask_bwd,
                'X_fwd': X_fwd,
                'weights_fwd': weights_fwd,
                'time_ids_fwd': time_ids_fwd,
                'batch_ids_fwd': tf.matrix_transpose(batch_ids_fwd),
                'mask_fwd': mask_fwd,
                'time_ids_at_pred': time_ids_at_pred,
                'batch_ids_at_pred': batch_ids_at_pred
            }

            return out


def align_values_to_batch(
        batch_batch_ids,
        batch_time_ids,
        value_batch_ids,
        value_time_ids
):
    i = 0 # batch counter
    j = 0 # value counter

    assert len(batch_batch_ids) == len(batch_time_ids), 'Batch and time IDs for batch must be vectors of the same size. Saw %s and %s.' % (len(batch_batch_ids), len(batch_time_ids))
    assert len(value_batch_ids) == len(value_time_ids), 'Batch and time IDs for memory must be vectors of the same size. Saw %s and %s.' % (len(value_batch_ids), len(value_time_ids))

    n_batch = len(batch_batch_ids)
    n_val = len(value_batch_ids)

    out = np.zeros_like(batch_batch_ids)
    dummy = n_val

    val_prev_b = -1
    val_prev_t = -1

    while i < n_batch or j < n_val:
        if i < n_batch and j < n_val:
            batch_b, batch_t = batch_batch_ids[i], batch_time_ids[i]
            val_b, val_t = value_batch_ids[j], value_time_ids[j]

            if batch_b < val_b:
                if batch_b == val_prev_b and batch_t >= val_prev_t:
                    out[i] = j - 1
                else: # Preceding value is either from a different batch or a later timestep
                    out[i] = dummy
                i += 1
            elif batch_b > val_b:
                val_prev_b = val_b
                val_prev_t = val_t
                j += 1
            else: # batch_b == val_b
                if batch_t == val_t:
                    out[i] = j
                    val_prev_b = val_b
                    val_prev_t = val_t
                    i += 1
                    j += 1
                elif batch_t < val_t:
                    if batch_b == val_prev_b and batch_t >= val_prev_t:
                        if j == 0:
                            out[i] = dummy
                        else: # j > 0
                            out[i] = j - 1
                    else: # Preceding value is either from a different batch or a later timestep
                        out[i] = dummy
                    i += 1
                else: # batch_t > val_t
                    val_prev_b = val_b
                    val_prev_t = val_t
                    j += 1
        elif i < n_batch:
            if j == 0:
                out[i] = dummy
            else: # j > 0
                out[i] = j - 1
            i += 1
        else: # i >= n_batch and j < n_val
            j += 1

    return out

def get_segment_indices(segs, mask):
    shape = segs.shape
    batch_ix = np.arange(shape[0])
    while len(batch_ix.shape) < len(shape):
        batch_ix = batch_ix[..., None]
    batch_ix = np.tile(batch_ix, np.floor_divide(shape, batch_ix.shape))
    batch_ix = batch_ix.reshape(-1)
    segs = segs.reshape(-1)
    mask = mask.reshape(-1)
    segs = segs > 0.5
    mask = mask > 0.5

    out_ix = []
    max_len = 0
    len_cur = 0
    ix_cur = []
    b_cur = -1

    for i, (s, m, b) in enumerate(zip(segs, mask, batch_ix)):
        if b != b_cur:
            ix_cur = []
            b_cur = b
        if m:
            ix_cur.append(i)
            len_cur += 1
            if s:
                out_ix.append(ix_cur)
                ix_cur = []
                if len_cur > max_len:
                    max_len = len_cur
                len_cur = 0

    out_mask_bottom = []

    for i, x in enumerate(out_ix):
        mask_cur = [1] * len(x)
        if len(x) < max_len:
            pad = [0] * (max_len - len(x))
            out_ix[i] = x + pad
            mask_cur += pad
        out_mask_bottom.append(mask_cur)
            
    out_ix = np.array(out_ix, dtype='int32')
    out_mask_bottom = np.array(out_mask_bottom, dtype='float32')

    return out_ix, out_mask_bottom


def update_binding_matrix(k, v, M=None, beta=0., epsilon=1e-8, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            dims = max([len(x.shape) for x in (k, v)])
            while len(k.shape) < dims:
                k = k[None, ...]
            while len(v.shape) < dims:
                v = v[None, ...]

            k_shape = tf.shape(k)
            v_shape = tf.shape(v)
            K = tf.shape(k_shape[-1])
            V = tf.shape(v_shape[-1])

            if beta is None:
                beta = 0
            if M is None:
                M = tf.zeros([K, V])

            k = tf.nn.l2_normalize(k, epsilon=epsilon, axis=-1)
            v = tf.nn.l2_normalize(v, epsilon=epsilon, axis=-1)
            P_denom = tf.maximum(tf.reduce_sum(k**2, axis=-1), epsilon)

            P = (k[..., None] * tf.expand_dims(k, -2)) / P_denom[..., None, None]
            _P = tf.eye(int(k.shape[-1]))
            while len(_P.shape) < len(P.shape):
                _P = _P[None, ...]
            _P -= P

            prev = tf.matmul(P, M)
            new = k[..., None] * tf.expand_dims(v, -2)
            out = tf.matmul(_P, M) + beta * prev + (1 - beta) * new

            return out


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
                    elif activation.lower() == 'round':
                        def out(x):
                            return tf.round(x)
                    elif activation.lower() == 'stop_gradient':
                        def out(x):
                            return tf.stop_gradient(x)
                    elif activation.lower() == 'argmax':
                        def out(x):
                            dim = x.shape[-1]
                            one_hot = tf.one_hot(tf.argmax(x, axis=-1), dim)
                            return one_hot
                    elif activation.lower() in ['bsn', 'csn']:
                        if activation.lower() == 'bsn':
                            sample_fn_inner = bernoulli_straight_through
                            round_fn_inner = round_straight_through
                            if from_logits:
                                logits2probs = tf.sigmoid
                            else:
                                logits2probs = lambda x: x
                        else: # activation.lower() == 'csn'
                            sample_fn_inner = argmax_straight_through
                            round_fn_inner = categorical_sample_straight_through
                            if from_logits:
                                logits2probs = tf.nn.softmax
                            else:
                                logits2probs = lambda x: x

                        def make_sample_fn(s, logit_fn=logits2probs, fn=sample_fn_inner):
                            def sample_fn(x):
                                return fn(logit_fn(x), session=s)

                            return sample_fn

                        def make_round_fn(s, logit_fn=logits2probs, fn=round_fn_inner):
                            def round_fn(x):
                                return fn(logit_fn(x), session=s)

                            return round_fn

                        sample_fn = make_sample_fn(session)
                        round_fn = make_round_fn(session)

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
                if init.lower() == 'l1_regularizer':
                    out = lambda x, scale=scale: tf.abs(x) * scale
                if init.lower() == 'l2_regularizer':
                    out = lambda x, scale=scale: x**2 * scale
                else:
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
            # bw_op = tf.identity
            def bw_op(x):
                out = tf.round(x)
                corr = tf.stop_gradient((0.5 - tf.abs(out - x)) / 0.5)
                out *= corr
                return out
            return replace_gradient(fw_op, bw_op, session=session)(x)


def bernoulli_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = lambda x: tf.ceil(x - tf.random_uniform(tf.shape(x)))
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)


def argmax_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = lambda x: tf.one_hot(tf.argmax(x, axis=-1), x.shape[-1])
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)


def categorical_sample_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = lambda x: tf.one_hot(tf.contrib.distributions.Categorical(probs=x).sample(), x.shape[-1])
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)


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
            if dim:
                embedding_matrix = tf.Variable(tf.fill([n_categories+1, dim], default), name=name)
            else:
                embedding_matrix = None

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
            filter /= tf.maximum(tf.reduce_sum(filter), 1e-8)
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


def infinity_mask(a, mask=None, float_type=tf.float32, int_type=tf.int32, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if mask is not None:
                mask = tf.cast(mask, dtype=tf.bool)
                while len(mask.shape) < len(a.shape):
                    mask = mask[..., None]
                tile_ix = tf.cast(tf.shape(a) / tf.maximum(tf.shape(mask), 1), dtype=int_type)
                mask = tf.tile(mask, tile_ix)
                alt = tf.fill(tf.shape(a), -float_type.max)
                a = tf.where(mask, a, alt)

            return a
        

def scaled_dot_attn(
        q,
        k,
        v,
        normalize_repeats=False,
        mask=None,
        float_type=tf.float32,
        int_type=tf.int32,
        epsilon=1e-8,
        session=None
):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            q = tf.expand_dims(q, axis=-2)
            scale = tf.sqrt(tf.cast(tf.shape(q)[-1], dtype=float_type))
            if normalize_repeats:
                ids = v * mask[..., None]
                counts = tf.matrix_transpose(tf.reduce_sum(ids, axis=-2, keepdims=True))
                weights = tf.matmul(ids, tf.reciprocal(tf.maximum(counts, epsilon)))
                k *= weights
            k = tf.matrix_transpose(k)
            dot = tf.squeeze(tf.matmul(q, k), axis=-2)
            dot = infinity_mask(dot, mask=mask, float_type=float_type, int_type=int_type, session=session)
            a = tf.nn.softmax(dot / tf.maximum(scale, epsilon))
            a_expanded = tf.expand_dims(a, -1)
            scaled = v * a_expanded
            out = tf.reduce_sum(scaled, axis=-2)

            return out, a


def reduce_losses(losses, weights=None, epsilon=1e-8, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if isinstance(losses, list):
                losses_tmp = []
                for i, l in enumerate(losses):
                    if weights is None or weights == 'normalize':
                        l_cur = tf.reshape(l, [-1])
                    elif isinstance(weights, list):
                        w = tf.convert_to_tensor(weights[i])
                        while len(w.shape) < len(l.shape):
                            w = w[..., None]
                        scale = tf.reduce_prod(tf.cast(tf.shape(l) / tf.maximum(tf.shape(w), 1), dtype=w.dtype))
                        l_cur = tf.reshape(l * w, [-1]) / tf.maximum(scale, epsilon)
                    else:
                        l_cur = tf.reshape(l * weights)
                    losses_tmp.append(l_cur)
                losses = tf.concat(losses_tmp, axis=0)
            else:
                losses = tf.reshape(losses, [-1])
            if weights is not None:
                if weights == 'normalize':
                    norm_const = tf.reduce_max(losses)
                    weights = losses / tf.maximum(norm_const, epsilon)
                    losses *= weights
                elif isinstance(weights, list):
                    weights = tf.concat([tf.reshape(w, [-1]) for w in weights], axis=0)
                else:
                    weights = tf.convert_to_tensor(weights)
                    while len(weights.shape) < len(losses.shape):
                        weights = weights[..., None]
                    scale = tf.reduce_prod(tf.cast(tf.shape(losses) / tf.maximum(tf.shape(weights), 1), dtype=weights.dtype))
                    weights *= scale

                losses = tf.reduce_sum(losses) / tf.maximum(tf.reduce_sum(weights), epsilon)
            else:
                losses = tf.reduce_mean(losses)

            return losses


HMLSTM_RETURN_SIGNATURE = [
    'c',
    'h',
    'features',
    'features_prob',
    'embedding',
    'features_by_seg',
    'feature_deltas',
    'z_prob',
    'z',
    'lm',
    'cell_proposal',
    'u',
    'v',
    'passthru',
    'M'
]


def hmlstm_state_size(
        units,
        layers,
        features=None,
        embedding=None,
        units_below=None,
        units_lm=None,
        units_cae=None,
        units_passthru=None,
        return_discrete=True,
        return_c=True,
        return_z_prob=True,
        return_lm=True,
        return_cell_proposal=True,
        return_M=True
):
    if return_lm:
        assert units_lm is not None, 'units_lm must be provided when using return_lm'
    if units_cae is not None:
        assert units_below is not None, 'units_below must be provided when using units_cae'
    size = []
    if isinstance(units, tuple):
        units = list(units)
    if not isinstance(units, list):
        units = [units] * layers
    if features is None:
        features = units
    if embedding is None:
        embedding = features
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
                # if l == 0 and units_passthru:
                if units_passthru:
                    value += units_passthru
            elif name == 'h':
                include = True
                value = units[l]
            elif name.startswith('feature'):
                if name in ['features_by_seg', 'feature_deltas']:
                    if l < layers - 1:
                        include = True
                        value = features[l]
                else:
                    include = True
                    value = features[l]
            elif name == 'embedding':
                include = True
                value = embedding[l]
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
                    # if l == 0 and units_passthru:
                    if units_passthru:
                        value += units_passthru
            elif name in ['u', 'v']:
                if units_cae[l] is not None and l < layers - 1:
                    include = True
                    if name == 'u':
                        value = 1
                    else:
                        value = units_cae[l]
            elif name == 'passthru':
                # if l == 0 and units_passthru:
                if units_passthru:
                    include = True
                    value = units_passthru
            elif name == 'M':
                if return_M and l < layers - 1:
                    include = True
                    value = embedding[l] ** 2
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
            num_features=None,
            num_embdims=None,
            training=False,
            neurons_per_boundary=1,
            boundary_neuron_agg_fn='logsumexp',
            neurons_per_feature=1,
            feature_neuron_agg_fn='logsumexp',
            cumulative_boundary_prob=False,
            cumulative_feature_prob=False,
            forget_at_boundary=True,
            recurrent_at_forget=False,
            renormalize_preactivations=False,
            append_previous_features=True,
            append_seg_len=True,
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
            boundary_noise_level=None,
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
            featurizer_regularizer=None,
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
            state_noise_level=None,
            feature_noise_level=None,
            bottomup_noise_level=None,
            recurrent_noise_level=None,
            topdown_noise_level=None,
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
            use_outer_product_memory=False,
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
                assert len(self._num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                if not isinstance(num_features, list):
                    self._num_features = [num_features] * num_layers
                else:
                    self._num_features = num_features
                assert len(self._num_features) == num_layers, 'num_features must either be a None, an int or a list of None/int of length num_layers'

                if not isinstance(num_embdims, list):
                    self._num_embdims = [num_embdims] * num_layers
                else:
                    self._num_embdims = num_embdims
                assert len(self._num_embdims) == num_layers, 'num_embdims must either be a None, an int or a list of None/int of length num_layers'

                self._num_layers = num_layers

                self._training = training

                self._neurons_per_boundary = neurons_per_boundary
                if boundary_neuron_agg_fn.lower() == 'uniform':
                    def bn_agg_fn(x, axis=-1, keepdims=False):
                        def train_fn(x):
                            b = tf.shape(x)[0]
                            ix = tf.random_uniform([b], 0, self._neurons_per_boundary, dtype=tf.int32)
                            ix = tf.stack([tf.range(b), ix], axis=-1)
                            out = tf.gather_nd(x, ix)
                            if keepdims:
                                out = tf.expand_dims(out, axis=axis)
                            return out

                        def eval_fn(x):
                            return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

                        if self._sample_at_eval:
                            return train_fn(x)
                        else:
                            return tf.cond(self._training, lambda: train_fn(x), lambda: eval_fn(x))
                else:
                    bn_agg_fn = getattr(tf, 'reduce_' + boundary_neuron_agg_fn)
                self._boundary_neuron_agg_fn = bn_agg_fn

                self._neurons_per_feature = neurons_per_feature
                if feature_neuron_agg_fn.lower() == 'uniform':
                    def fn_agg_fn(x, axis=-1, keepdims=False):
                        def train_fn(x):
                            b = tf.shape(x)[0]
                            f = x.shape[1]
                            ix = tf.random_uniform([b, f], 0, self._neurons_per_feature, dtype=tf.int32)
                            ix = tf.stack(
                                [
                                    tf.tile(
                                        tf.range(b)[..., None],
                                        [1, f]
                                    ),
                                    tf.tile(
                                        tf.range(f)[None, ...],
                                        [b, 1]
                                    ),
                                    ix
                                ],
                                axis=-1
                            )
                            out = tf.gather_nd(x, ix)
                            if keepdims:
                                out = tf.expand_dims(out, axis=axis)
                            return out

                        def eval_fn(x):
                            return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

                        if self._sample_at_eval:
                            return train_fn(x)
                        else:
                            return tf.cond(self._training, lambda: train_fn(x), lambda: eval_fn(x))
                else:
                    fn_agg_fn = getattr(tf, 'reduce_' + feature_neuron_agg_fn)
                self._feature_neuron_agg_fn = fn_agg_fn

                self._cumulative_boundary_prob = cumulative_boundary_prob
                self._cumulative_feature_prob = cumulative_feature_prob
                self._forget_at_boundary = forget_at_boundary
                self._recurrent_at_forget = recurrent_at_forget
                self._renormalize_preactivations = renormalize_preactivations
                self._append_previous_features = append_previous_features
                self._append_seg_len = append_seg_len

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

                self._activation = get_activation(
                    activation,
                    session=self._session,
                    training=self._training,
                    from_logits=True,
                    sample_at_train=sample_at_train,
                    sample_at_eval=sample_at_eval
                )
                self._tanh_activation = activation == 'tanh'
                self._inner_activation = get_activation(
                    inner_activation,
                    session=self._session,
                    training=self._training,
                    from_logits=True,
                    sample_at_train=sample_at_train,
                    sample_at_eval=sample_at_eval
                )
                self._tanh_inner_activation = inner_activation == 'tanh'
                self._prefinal_activation =  get_activation(
                    prefinal_activation,
                    session=self._session,
                    training=self._training,
                    from_logits=True,
                    sample_at_train=sample_at_train,
                    sample_at_eval=sample_at_eval
                )
                self._recurrent_activation = get_activation(
                    recurrent_activation,
                    session=self._session,
                    training=self._training,
                    from_logits=True,
                    sample_at_train=sample_at_train,
                    sample_at_eval=sample_at_eval
                )
                self._boundary_activation = get_activation(
                    boundary_activation,
                    session=self._session,
                    training=self._training,
                    from_logits=True,
                    sample_at_train=sample_at_train,
                    sample_at_eval=sample_at_eval
                )
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
                self._boundary_noise_level = boundary_noise_level

                self._bottomup_initializer = get_initializer(bottomup_initializer, session=self._session)
                self._recurrent_initializer = get_initializer(recurrent_initializer, session=self._session)
                self._topdown_initializer = get_initializer(topdown_initializer, session=self._session)
                self._boundary_initializer = get_initializer(boundary_initializer, session=self._session)
                self._bias_initializer = get_initializer(bias_initializer, session=self._session)

                self._bottomup_regularizer = get_regularizer(bottomup_regularizer, session=self._session)
                self._recurrent_regularizer = get_regularizer(recurrent_regularizer, session=self._session)
                self._topdown_regularizer = get_regularizer(topdown_regularizer, session=self._session)
                self._boundary_regularizer = get_regularizer(boundary_regularizer, session=self._session)
                self._featurizer_regularizer = get_regularizer(featurizer_regularizer, session=self._session)
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
                self._state_noise_level = state_noise_level
                self._feature_noise_level = feature_noise_level
                self._bottomup_noise_level = bottomup_noise_level
                self._recurrent_noise_level = recurrent_noise_level
                self._topdown_noise_level = topdown_noise_level
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

                self._use_outer_product_memory = use_outer_product_memory

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
                    elif 'featurizer' in n1:
                        self._add_regularization(var, self._featurizer_regularizer)
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

        n_features = []
        for l in range(self._num_layers):
            if self._num_features[l] is None:
                n_features.append(self._num_units[l])
            else:
                n_features.append(self._num_features[l])

        n_embdims = []
        for l in range(self._num_layers):
            if self._num_embdims[l] is None:
                n_embdims.append(n_features[l])
            else:
                n_embdims.append(self._num_embdims[l])

        units_cae = []
        for l in range(self._num_layers):
            if l == 0:
                units_cae.append(self._input_dims)
            else:
                if self._num_features[l-1] is None:
                    units_cae.append(self._num_units[l-1])
                else:
                    units_cae.append(self._num_features[l-1])

        return hmlstm_state_size(
            self._num_units,
            self._num_layers,
            features=n_features,
            embedding=n_embdims,
            units_below=units_below,
            units_lm=units_lm,
            units_cae=units_cae,
            units_passthru=self._n_passthru_neurons,
            return_lm=self._lm,
            return_c=True,
            return_z_prob=True,
            return_cell_proposal=True,
            return_M=self._use_outer_product_memory
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

    def initialize_kernel(
            self,
            l,
            in_dim,
            out_dim,
            kernel_initializer,
            depth=None,
            prefinal_mode=None,
            final_bias=False,
            name=None
    ):
        with self._session.as_default():
            with self._session.graph.as_default():
                units_below = in_dim
                kernel_lambdas = []
                if depth is None:
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
                        use_bias = final_bias
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

                    if self._resnet_n_layers and self._resnet_n_layers > 1 and units == units_below:
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

                    units_below = units

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
                self._kernel_featurizer = []
                self._kernel_embedding = []
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

                if self._use_outer_product_memory:
                    self._key_val_maps = []

                if self._oracle_boundary:
                    n_boundary_dims = self._num_layers - 1
                else:
                    n_boundary_dims = 0

                for l in range(self._num_layers):
                    if l == 0:
                        bottomup_dim = inputs_shape[1].value - n_boundary_dims
                    elif self._num_features[l - 1] is None:
                        bottomup_dim = self._num_units[l - 1]
                    else:
                        bottomup_dim = self._num_features[l - 1]
                    bottomup_dim += (self._implementation == 1)
                    if l > 0:
                        bottomup_dim += self._append_seg_len

                    recurrent_dim = self._num_units[l]
                    if self._append_previous_features and self._num_features[l]:
                        recurrent_dim += 2 * self._num_features[l]

                    output_dim = self._num_units[l]
                    # if l == 0 and self._n_passthru_neurons:
                    if self._n_passthru_neurons:
                        output_dim += self._n_passthru_neurons
                    output_dim *= 4 # forget, input, and output gates, plus cell proposal
                    if self._implementation == 1:
                        if l < self._num_layers - 1:
                            output_dim += self._neurons_per_boundary

                    # Build bias
                    if not self._layer_normalization and self._use_bias:
                        bias = self.add_variable(
                            'bias_l%d' % l,
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
                        if self._num_features[l + 1] is None:
                            topdown_dim = self._num_units[l + 1]
                        else:
                            topdown_dim = self._num_features[l + 1]
                        topdown_dim += (self._implementation == 1)

                        self._kernel_topdown.append(
                            self.initialize_kernel(
                                l,
                                topdown_dim,
                                output_dim,
                                self._topdown_initializer,
                                name='topdown'
                            )
                        )

                    # Build featurizer kernel
                    if self._num_features[l] is not None:
                        featurizer_in_dim = self._num_units[l]
                        n_feats = self._num_features[l]
                        kernel_featurizer = self.initialize_kernel(
                            l,
                            featurizer_in_dim,
                            n_feats * self._neurons_per_feature,
                            self._bottomup_initializer,
                            name='featurizer'
                        )
                    else:
                        kernel_featurizer = lambda x: x
                    self._kernel_featurizer.append(kernel_featurizer)

                    # Build embedding kernel
                    if self._num_embdims[l] is not None:
                        if self._num_features[l]:
                            n_feats = self._num_features[l]
                        else:
                            n_feats = self._num_units[l]
                        kernel_embedding = self.initialize_kernel(
                            l,
                            n_feats,
                            self._num_embdims[l],
                            self._bottomup_initializer,
                            name='embedding'
                        )
                    else:
                        kernel_embedding = lambda x: x
                    if self._use_outer_product_memory:
                        key_val_map = self.initialize_kernel(
                            l,
                            self._num_units[l],
                            self._num_units[l],
                            self._bottomup_initializer,
                            name='key_val_map'
                        )
                        self._key_val_maps.append(key_val_map)

                        def retrieve(x, d):
                            out = self._key_val_maps[d](x)
                            out = tf.nn.l2_normalize(out, epsilon=self._epsilon, axis=-1)

                            return out

                        def get_embd_fn(retrieve_fn=retrieve, d=l, emdb_fn=kernel_embedding):
                            def fn(x):
                                out = retrieve_fn(x, d)
                                out = emdb_fn(out)

                                return out
                            return fn

                        kernel_embedding = get_embd_fn()

                    self._kernel_embedding.append(kernel_embedding)

                    # Build boundary (and, if necessary, boundary kernel)
                    if l < self._num_layers - 1 and self._infer_boundary:
                        if self._implementation == 1 and not self._layer_normalization and self._use_bias:
                            bias_boundary = bias[:, -self._neurons_per_boundary:]
                            self._bias_boundary.append(bias_boundary)
                        elif self._implementation == 2:
                            boundary_in_dim = self._num_units[l]
                            if self._num_features[l] is not None:
                                boundary_in_dim += self._num_features[l]

                            self._kernel_boundary.append(
                                self.initialize_kernel(
                                    l,
                                    boundary_in_dim,
                                    self._neurons_per_boundary,
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
                                # if self._n_passthru_neurons: # Alternates with following two lines
                                #     lm_bottomup_in_dim += self._n_passthru_neurons
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
                                # if self._n_passthru_neurons: # Alternates with following two lines
                                #     lm_in_dim += self._n_passthru_neurons
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
                    input_feats = h_below
                else:
                    h_below = inputs
                    input_feats = h_below

                z_below = None

                for l, layer in enumerate(state):
                    h_below_clean = h_below
                    h_below = self._bottomup_dropout(h_below)

                    # if not self._bptt:
                    #     h_below = tf.stop_gradient(h_below)

                    if self._revnet_n_layers:
                        h_below = self._revnet[l].forward(h_below)

                    if self._bottomup_noise_level:
                        h_below = tf.cond(
                            self._training,
                            lambda: h_below + tf.random_normal(shape=tf.shape(h_below), stddev=self._bottomup_noise_level),
                            lambda: h_below
                        )

                    # z_behind: Previous boundary probability at current layer (implicitly 1 if final layer)
                    if l < self._num_layers - 1:
                        z_behind = layer.z
                        z_prob_behind = layer.z_prob
                    else:
                        z_behind = None
                        z_prob_behind = None

                    # Compute probability of update, copy, and flush operations operations.
                    if z_behind is None:
                        z_behind_cur = 0
                    else:
                        z_behind_cur = z_behind
                    if z_prob_behind is None:
                        z_prob_behind_cur = 0
                    else:
                        z_prob_behind_cur = z_prob_behind

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
                    # if l == 0 and self._n_passthru_neurons:
                    if self._n_passthru_neurons:
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
                    if self._recurrent_noise_level:
                        h_behind = tf.cond(
                            self._training,
                            lambda: h_behind + tf.random_normal(shape=tf.shape(h_behind), stddev=self._recurrent_noise_level),
                            lambda: h_behind
                        )
                    h_behind = self._recurrent_dropout(h_behind)

                    # Previous features
                    features_behind = layer.features
                    features_prob_behind = layer.features_prob

                    # h_above: Hidden state of layer above at previous timestep (implicitly 0 if final layer)
                    if l < self._num_layers - 1:
                        h_above = state[l + 1].embedding
                        # h_above = state[l + 1].h

                        # if not self._bptt:
                        #     h_above = tf.stop_gradient(h_above)
                        if self._topdown_noise_level:
                            h_above = tf.cond(
                                self._training,
                                lambda: h_above + tf.random_normal(shape=tf.shape(h_above), stddev=self._topdown_noise_level),
                                lambda: h_above
                            )
                        h_above = self._topdown_dropout(h_above)
                    else:
                        h_above = None

                    # Compute state preactivations
                    s = []

                    bottomup_in = h_below
                    if l > 0 and self._append_seg_len:
                        bottomup_in = tf.concat([bottomup_in, u], axis=-1)

                    s_bottomup = self._kernel_bottomup[l](bottomup_in)
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
                    elif l > 0:
                        normalizer = z_below
                    else:
                        normalizer = 1.

                    s.append(s_bottomup)

                    # Recurrent features
                    recurrent_in = h_behind
                    if self._append_previous_features and l < self._num_layers - 1:
                        recurrent_in = tf.concat([recurrent_in, layer.features, layer.features_by_seg], axis=-1)
                    s_recurrent = self._kernel_recurrent[l](recurrent_in)
                    if self._forget_at_boundary and not self._recurrent_at_forget:
                        p = (1 - flush_prob)
                        s_recurrent *= p
                        normalizer += p
                    else:
                        normalizer += 1.
                    s.append(s_recurrent)

                    # Top-down features (if non-final layer)
                    if l < self._num_layers - 1:
                        # Compute top-down features
                        s_topdown = self._kernel_topdown[l](h_above) * z_behind
                        normalizer += z_behind
                        # Add in top-down features
                        s.append(s_topdown)

                    s = sum(s)

                    if self._renormalize_preactivations:
                        normalizer += self._epsilon
                        s /= normalizer

                    s_clean = s
                    if self._state_noise_level:
                        s = tf.cond(self._training, lambda: s + tf.random_normal(shape=tf.shape(s), stddev=self._state_noise_level), lambda: s)

                    if self._implementation == 1:
                        # In implementation 1, boundary has its own slice of the hidden state preactivations
                        z_logit = s[:, units * 4:]
                        s = s[:, :units * 4]
                        s_clean = s_clean[:, :units * 4]
                    else:
                        z_logit = None

                    if not self._layer_normalization and self._use_bias:
                        if self._implementation == 1:
                            z_logit += self._bias[l][:, units * 4:]
                        s = s + self._bias[l][:, :units * 4]
                        s_clean = s_clean + self._bias[l][:, :units * 4]

                    # Forget gate
                    f = s[:, :units]
                    f_clean = s_clean[:, :units]
                    if self._layer_normalization:
                        f = self.norm(f, 'f_ln_%d' % l)
                        f_clean = self.norm(f_clean, 'f_ln_%d' % l)
                    f = self._recurrent_activation(f + self._forget_bias)
                    f_clean = self._recurrent_activation(f_clean + self._forget_bias)


                    # Input gate
                    i = s[:, units:units * 2]
                    i_clean = s_clean[:, units:units * 2]
                    if self._layer_normalization:
                        i = self.norm(i, 'i_ln_%d' % l)
                        i_clean = self.norm(i_clean, 'i_ln_%d' % l)
                    i = self._recurrent_activation(i)
                    i_clean = self._recurrent_activation(i_clean)

                    # Output gate
                    o = s[:, units * 2:units * 3]
                    o_clean = s_clean[:, units * 2:units * 3]
                    if self._layer_normalization:
                        o = self.norm(o, 'o_ln_%d' % l)
                        o_clean = self.norm(o_clean, 'o_ln_%d' % l)
                    o = self._recurrent_activation(o)
                    o_clean = self._recurrent_activation(o_clean)

                    # Cell proposal
                    g = s[:, units * 3:units * 4]
                    g_clean = s_clean[:, units * 3:units * 4]
                    if self._layer_normalization:
                        g = self.norm(g, 'g_ln_%d' % l)
                        g_clean = self.norm(g_clean, 'g_ln_%d' % l)
                    g = activation(g)
                    g_clean = activation(g_clean)

                    # Compute cell state flush operation
                    c_flush = i * g
                    c_flush_clean = i_clean * g_clean

                    # Compute cell state copy operation
                    c_copy = c_behind

                    # Cell state update (forget-gated previous cell plus input-gated cell proposal)
                    c_update = f * c_behind + c_flush
                    c_update_clean = f_clean * c_behind + c_flush_clean

                    # Merge cell operations. If boundaries are hard, selects between update, copy, and flush.
                    # If boundaries are soft, sums update copy and flush proportionally to their probs.
                    if self._forget_at_boundary:
                        c = update_prob * c_update + flush_prob * c_flush + copy_prob * c_copy
                        c_clean = update_prob * c_update_clean + flush_prob * c_flush_clean + copy_prob * c_copy
                    else:
                        c = z_below_cur * c_update + (1 - z_below_cur) * c_copy
                        c_clean = z_below_cur * c_update_clean + (1 - z_below_cur) * c_copy

                    # Compute the gated output of non-copy cell state
                    h = c
                    h_clean = c_clean
                    if self._state_discretizer and self._state_slope_annealing_rate and self._global_step is not None:
                        h *= self.state_slope_coef
                        h_clean *= self.state_slope_coef
                    h = activation(h) * o
                    h_clean = activation(h_clean) * o_clean

                    # if l == 0 and self._n_passthru_neurons:
                    if self._n_passthru_neurons:
                        passthru = h[:, :self._n_passthru_neurons]
                        passthru_clean = h_clean[:, :self._n_passthru_neurons]
                        h = h[:, self._n_passthru_neurons:]
                        h_clean = h_clean[:, self._n_passthru_neurons:]
                    else:
                        passthru = None
                        passthru_clean = None

                    if self._l2_normalize_states or (l < self._num_layers - 1 and self._use_outer_product_memory):
                        h = tf.nn.l2_normalize(h, epsilon=self._epsilon, axis=-1)
                        h_clean = tf.nn.l2_normalize(h_clean, epsilon=self._epsilon, axis=-1)

                    if self._num_features[l] is None and not self._use_outer_product_memory:
                        # Squash to [-1,1] if state activation is not tanh
                        if (l < self._num_layers - 1 and not self._tanh_inner_activation) \
                                or (l == self._num_layers - 1 and not self._tanh_activation):
                            h_prob = tf.tanh(h)
                            h_prob_clean = tf.tanh(h_clean)
                        else:
                            h_prob = h
                            h_prob_clean = h_clean

                        # State probabilities in [0,1]
                        h_prob = (h_prob + 1) / 2
                        h_prob_clean = (h_prob_clean + 1) / 2
                        if self._state_discretizer:
                            # State values in {0,1}
                            h_discrete = self._state_discretizer(h_prob)
                            h_discrete_clean = round_straight_through(h_prob_clean, session=self._session)

                            if self._min_discretization_prob is not None:
                                def train_func(h_discrete=h_discrete, h_prob=h_prob, h=h):
                                    a = self._min_discretization_prob
                                    discretization_prob = 2 * (1 - a) * tf.abs(h_prob - 0.5) + a
                                    if self._trainable_self_discretization:
                                        discretize = self._state_discretizer(discretization_prob)
                                        out = h_discrete * discretize + h * (1 - discretize)
                                    else:
                                        discretize = tf.random_uniform(
                                            tf.shape(discretization_prob)) < discretization_prob
                                        out = tf.where(discretize, h_discrete, h)
                                    return out

                                def eval_func(h_discrete=h_discrete):
                                    return h_discrete

                                h_discrete = tf.cond(self._training, train_func, eval_func)
                                    
                            if not self._discretize_state_at_boundary and l < (self._num_layers - 1 + self._discretize_final):
                                h = h_discrete
                                h_clean = h_discrete_clean

                    # Mix gated output with the previous hidden state proportionally to the copy probability
                    if self._forget_at_boundary: # Copy depends on both recurrent and bottom-up boundary probs
                        h = copy_prob * h_behind + (1 - copy_prob) * h
                        h_clean = copy_prob * h_behind + (1 - copy_prob) * h_clean
                    else: # Copy just depends on bottom-up boundary probs
                        h = (1 - z_below_cur) * h_behind + z_below_cur * h
                        h_clean = (1 - z_below_cur) * h_behind + z_below_cur * h_clean

                    if self._num_features[l] is None:
                        if self._state_discretizer and l < (self._num_layers - 1 + self._discretize_final):
                            features_discrete = h_discrete
                            features_discrete_clean = h_discrete_clean
                            features = features_discrete
                            features_clean = features_discrete_clean
                        else:
                            features = h
                            features_clean = h_clean
                        features_prob = (h + 1) / 2
                    else:
                        # Define features
                        features_in = h
                        features_in_clean = h_clean
                        features = self._kernel_featurizer[l](features_in)
                        features_clean = self._kernel_featurizer[l](features_in_clean)

                        if self._feature_noise_level:
                            features = tf.cond(
                                self._training,
                                lambda: features + tf.random_normal(shape=tf.shape(features), stddev=self._feature_noise_level),
                                lambda: features
                            )

                        if self._neurons_per_feature > 1:
                            features = tf.reshape(features, [-1, self._num_features[l], self._neurons_per_feature])
                            features_clean = tf.reshape(features_clean, [-1, self._num_features[l], self._neurons_per_feature])

                            features = self._feature_neuron_agg_fn(features, axis=-1, keepdims=False)
                            features_clean = self._feature_neuron_agg_fn(features_clean, axis=-1, keepdims=False)

                        # Feature values in {0,1}
                        features_prob = self._recurrent_activation(features[..., :self._num_features[l]])

                        if self._state_discretizer:
                            # Feature probabilities in [0,1]

                            if self._cumulative_feature_prob:
                                feat_prob_prev = layer.features_prob * (1 - z_behind_cur)
                                feat_prob_range = 1 - feat_prob_prev
                                feat_prob_update = feat_prob_prev + features_prob * feat_prob_range
                                if self._forget_at_boundary:
                                    cprob = copy_prob
                                else:
                                    cprob = z_below_cur
                                features_prob = (1 - cprob) * feat_prob_prev + cprob * feat_prob_update

                            # Feature values in {0,1}
                            features_discrete = self._state_discretizer(features_prob)
                            features_discrete_clean = round_straight_through(features_prob, session=self._session)

                            if l < (self._num_layers - 1 + self._discretize_final):
                                if self._min_discretization_prob is not None:
                                    def train_func(features_discrete=features_discrete, features_prob=features_prob, features=features):
                                        a = self._min_discretization_prob
                                        discretization_prob = 2 * (1 - a) * tf.abs(features_prob - 0.5) + a
                                        if self._trainable_self_discretization:
                                            discretize = self._state_discretizer(discretization_prob)
                                            out = features_discrete * discretize + features * (1 - discretize)
                                        else:
                                            discretize = tf.random_uniform(
                                                tf.shape(discretization_prob)) < discretization_prob
                                            out = tf.where(discretize, features_discrete, features)
                                        return out
    
                                    def eval_func(features_discrete=features_discrete):
                                        return features_discrete
    
                                    features_discrete = tf.cond(self._training, train_func, eval_func)

                                features = features_discrete
                                features_clean = features_discrete_clean
                        elif not self._use_outer_product_memory:
                            features = features_prob
                            features_clean = features_prob

                    if self._num_features[l]:
                        if self._forget_at_boundary:
                            features = copy_prob * features_behind + (1 - copy_prob) * features
                            features_prob = copy_prob * features_prob_behind + (1 - copy_prob) * features_prob
                        else: # Copy just depends on bottom-up boundary probs
                            features = (1 - z_below_cur) * features_behind + z_below_cur * features
                            features_prob = (1 - z_below_cur) * features_prob_behind + z_below_cur * features_prob

                    embedding = self._kernel_embedding[l](features)

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
                                if self._num_features[l] is not None:
                                    if self._num_embdims:
                                        feats = embedding
                                    else:
                                        feats = features
                                    z_in.append(feats)

                                if len(z_in) == 1:
                                    z_in = z_in[0]
                                else:
                                    z_in = tf.concat(z_in, axis=1)
                                z_in = self._boundary_dropout(z_in)
                                # In implementation 2, boundary is a function of the hidden state
                                z_logit = self._kernel_boundary[l](z_in)

                            if self._boundary_discretizer and self._boundary_slope_annealing_rate and self._global_step is not None:
                                z_logit *= self.boundary_slope_coef

                            if self._boundary_noise_level:
                                z_logit = tf.cond(self._training, lambda: z_logit + tf.random_normal(shape=tf.shape(z_logit), stddev=self._boundary_noise_level), lambda: z_logit)

                            if self._neurons_per_boundary > 1:
                                z_logit = self._boundary_neuron_agg_fn(z_logit, axis=-1, keepdims=True)

                            z_prob = self._boundary_activation(z_logit)

                            if self._cumulative_boundary_prob:
                                z_prob_prev = z_prob_behind_cur * (1 - z_behind_cur)
                                z_prob_range = 1 - z_prob_prev
                                z_prob_update = z_prob_prev + z_prob * z_prob_range
                                if self._forget_at_boundary:
                                    cprob = copy_prob
                                else:
                                    cprob = (1 - z_below_cur)
                                z_prob = cprob * z_prob_prev + (1 - cprob) * z_prob_update

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
                        features_by_seg = features * z + layer.features_by_seg * (1 - z)
                        feature_deltas = features_by_seg - layer.features_by_seg
                        feature_deltas = tf.maximum(-feature_deltas, feature_deltas)
                    else:
                        z_prob = None
                        z = None
                        features_by_seg = None
                        feature_deltas = None

                    if l < self._num_layers - 1:
                        # Compute segment lengths and averaged segment values
                        u_behind = state[l].u
                        v_behind = state[l].v

                        # Update segment length
                        u = u_behind * (1 - z_behind_cur) + z_below_cur

                        # Update mean segment features
                        if l == 0:
                            feats = input_feats
                        else:
                            feats = state[l-1].features

                        cond = tf.squeeze(u > 0, axis=-1)

                        v = tf.where(
                            cond,
                            (v_behind * tf.maximum(u - 1, 0) + feats * z_below_cur) / tf.where(cond, u, tf.ones_like(u)), # The redundant use of tf.where is needed here because of a bug that backprops NaNs from the false branch even though it isn't selected
                            tf.zeros_like(v_behind)
                        )

                        # Update binding matrix M
                        if self._use_outer_product_memory:
                            key = features
                            val = embedding
                            M_prev = layer.M
                            M_prev_shape = tf.shape(M_prev)
                            final_dim = int(M_prev.shape[-1])
                            b = []
                            for i in range(len(M_prev.shape) - 1):
                                b.append(M_prev_shape[i])
                            side_len = tf.cast(tf.sqrt(tf.cast(final_dim, dtype=tf.float32)), dtype=tf.int32)
                            M_prev = tf.reshape(M_prev, b + [side_len, side_len])
                            M_new = update_binding_matrix(key, val, M=M_prev, beta=0., epsilon=self._epsilon, session=self._session)
                            M = M_new * z[..., None] + M_prev * (1 - z)[..., None]
                            embedding = tf.squeeze(tf.matmul(tf.expand_dims(key, -2), M), axis=-2)
                            M = tf.reshape(M, b + [final_dim])
                        else:
                            M = None
                    else:
                        v = u = M = None

                    name_map = {
                        'c': c,
                        'h': h,
                        'features': features,
                        'features_prob': features_prob,
                        'features_by_seg': features_by_seg,
                        'feature_deltas': feature_deltas,
                        'embedding': embedding,
                        'z': z,
                        'z_prob': z_prob,
                        'cell_proposal': c_flush,
                        'passthru': passthru,
                        'v': v,
                        'u': u,
                        'M': M
                    }

                    h_below = embedding
                    z_below = z

                    name_maps.append(name_map)

                # Compute any required predictions now that entire encoder state is available
                new_state = []
                for l in range(self._num_layers - 1, -1, -1):
                    name_map = name_maps[l]
                    h = name_map['embedding']
                    z = name_map['z']
                    passthru = name_map['passthru']

                    if self._lm:
                        if self._lm_use_upper:
                            lm_h_in = [s['embedding'] for s in name_maps[l:]]
                            if len(lm_h_in) > 1:
                                lm_h_in = tf.concat(lm_h_in, axis=-1)
                            else:
                                lm_h_in = lm_h_in[0]
                        else:
                            lm_h_in = h

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

                    new_state_names = []
                    new_state_cur = []

                    for name in HMLSTM_RETURN_SIGNATURE:
                        include = False
                        if name in ['c', 'h', 'features', 'features_prob', 'embedding', 'cell_proposal']:
                            include = True
                        elif name in ['z', 'z_prob'] and l < self._num_layers - 1:
                            include = True
                        elif self._lm and name == 'lm':
                            include = True
                        elif l < self._num_layers - 1 and name in ['u', 'v']:
                            include = True
                        # elif self._n_passthru_neurons and l == 0 and name == 'passthru':
                        elif self._n_passthru_neurons and name == 'passthru':
                            include = True
                        elif l < self._num_layers - 1 and name in ['features_by_seg', 'feature_deltas']:
                            include = True
                        elif name == 'M' and name_map[name] is not None:
                            include = True

                        if include and name in name_map:
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
            num_features=None,
            num_embdims=None,
            training=False,
            neurons_per_boundary=1,
            boundary_neuron_agg_fn='logsumexp',
            neurons_per_feature=1,
            feature_neuron_agg_fn='logsumexp',
            cumulative_boundary_prob=False,
            cumulative_feature_prob=False,
            forget_at_boundary=True,
            recurrent_at_forget=False,
            renormalize_preactivations=False,
            append_previous_features=True,
            append_seg_len=True,
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
            boundary_noise_level=None,
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            topdown_initializer='glorot_uniform_initializer',
            boundary_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            topdown_regularizer=None,
            boundary_regularizer=None,
            featurizer_regularizer=None,
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
            state_noise_level=None,
            feature_noise_level=None,
            bottomup_noise_level=None,
            recurrent_noise_level=None,
            topdown_noise_level=None,
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
            use_outer_product_memory=None,
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

                if not isinstance(num_features, list):
                    self.num_features = [num_features] * num_layers
                else:
                    self.num_features = num_features
                assert len(self.num_features) == num_layers, 'num_features must either be None, an int or a list of None/int of length num_layers'

                if not isinstance(num_embdims, list):
                    self.num_embdims = [num_embdims] * num_layers
                else:
                    self.num_embdims = num_embdims
                assert len(self.num_embdims) == num_layers, 'num_embdims must either be None, an int or a list of None/int of length num_layers'

                self.num_layers = num_layers
                self.training = training
                self.neurons_per_feature = neurons_per_boundary
                self.boundary_neuron_agg_fn = boundary_neuron_agg_fn
                self.num_feature_neurons = neurons_per_feature
                self.feature_neuron_agg_fn = feature_neuron_agg_fn
                self.cumulative_boundary_prob = cumulative_boundary_prob
                self.cumulative_feature_prob = cumulative_feature_prob
                self.forget_at_boundary = forget_at_boundary
                self.recurrent_at_forget = recurrent_at_forget
                self.renormalize_preactivations = renormalize_preactivations
                self.append_previous_features = append_previous_features
                self.append_seg_len = append_seg_len
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
                self.boundary_noise_level = boundary_noise_level

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
                self.featurizer_regularizer = featurizer_regularizer
                self.bias_regularizer = bias_regularizer

                self.weight_normalization = weight_normalization
                self.layer_normalization = layer_normalization
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
                self.state_noise_level = state_noise_level
                self.feature_noise_level = feature_noise_level
                self.bottomup_noise_level = bottomup_noise_level
                self.recurrent_noise_level = recurrent_noise_level
                self.topdown_noise_level = topdown_noise_level
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
                self.use_outer_product_memory = use_outer_product_memory
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
                    num_features=self.num_features,
                    num_embdims=self.num_embdims,
                    training=self.training,
                    neurons_per_boundary=self.neurons_per_feature,
                    boundary_neuron_agg_fn=self.boundary_neuron_agg_fn,
                    neurons_per_feature=self.num_feature_neurons,
                    feature_neuron_agg_fn=self.feature_neuron_agg_fn,
                    cumulative_boundary_prob=self.cumulative_boundary_prob,
                    cumulative_feature_prob=self.cumulative_feature_prob,
                    forget_at_boundary=self.forget_at_boundary,
                    recurrent_at_forget=self.recurrent_at_forget,
                    renormalize_preactivations=self.renormalize_preactivations,
                    append_previous_features=self.append_previous_features,
                    append_seg_len=self.append_seg_len,
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
                    boundary_noise_level=self.boundary_noise_level,
                    bottomup_initializer=self.bottomup_initializer,
                    recurrent_initializer=self.recurrent_initializer,
                    topdown_initializer=self.topdown_initializer,
                    bias_initializer=self.bias_initializer,
                    bottomup_regularizer=self.bottomup_regularizer,
                    recurrent_regularizer=self.recurrent_regularizer,
                    topdown_regularizer=self.topdown_regularizer,
                    boundary_regularizer=self.boundary_regularizer,
                    featurizer_regularizer=self.featurizer_regularizer,
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
                    state_noise_level=self.state_noise_level,
                    feature_noise_level=self.feature_noise_level,
                    bottomup_noise_level=self.bottomup_noise_level,
                    recurrent_noise_level=self.recurrent_noise_level,
                    topdown_noise_level=self.topdown_noise_level,
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
                    use_outer_product_memory=self.use_outer_product_memory,
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

                self.embedding_fn = self.cell._kernel_embedding

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

    def feature_vectors(self, level=None, mask=None):
        if level is None:
            out = tuple([l.feature_vectors(mask=mask) for l in self.l])
        else:
            out = self.l[level].feature_vectors(mask=mask)

        return out

    def embedding_vectors(self, level=None, mask=None):
        if level is None:
            out = tuple([l.embedding_vectors(mask=mask) for l in self.l])
        else:
            out = self.l[level].embedding_vectors(mask=mask)

        return out
    
    def feature_vectors_by_seg(self, level=None, mask=None):
        if level is None:
            out = tuple([l.feature_vectors_by_seg(mask=mask) for l in self.l[:-1]])
        else:
            out = self.l[level].feature_vectors_by_seg(mask=mask)

        return out
    
    def feature_delta_vectors(self, level=None, mask=None):
        if level is None:
            out = tuple([l.feature_delta_vectors(mask=mask) for l in self.l[:-1]])
        else:
            out = self.l[level].feature_delta_vectors(mask=mask)

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
                # return self.l[0].passthru_neurons(mask=mask)

                passthru_neurons = [l.passthru_neurons(mask=mask) for l in self.l]
                return passthru_neurons

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

    def feature_vectors(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.features

                if mask is not None:
                    while len(mask.shape) < len(out.shape):
                        mask = mask[..., None]
                    out = out * mask

                return out

    def embedding_vectors(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.embedding

                if mask is not None:
                    while len(mask.shape) < len(out.shape):
                        mask = mask[..., None]
                    out = out * mask

                return out

    def feature_vectors_by_seg(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.features_by_seg

                if mask is not None:
                    while len(mask.shape) < len(out.shape):
                        mask = mask[..., None]
                    out = out * mask

                return out

    def feature_delta_vectors(self, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.feature_deltas

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
            l2_normalize_states=False,
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

                self._l2_normalize_states = l2_normalize_states

                self._epsilon = epsilon

                self._regularizer_map = {}
                self._regularization_initialized = False

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(c=self._num_units,h=self._num_units)

    @property
    def output_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(c=self._num_units,h=self._num_units)

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
                units_below = in_dim
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

                    if self._resnet_n_layers and self._resnet_n_layers > 1 and units == units_below:
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

                    units_below = units

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

                if self._l2_normalize_states:
                    h = tf.nn.l2_normalize(h, epsilon=self._epsilon, axis=-1)

                if mask is not None:
                    c = c * mask + c_prev * (1 - mask)
                    h = h * mask + h_prev * (1 - mask)

                return tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h), tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)


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
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

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
            sample_at_train=False,
            sample_at_eval=False,
            batch_normalization_decay=None,
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
        self.activation = get_activation(
            activation,
            session=self.session,
            training=self.training,
            from_logits=True,
            sample_at_train=sample_at_train,
            sample_at_eval=sample_at_eval
        )
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
                sample_at_train=False,
                sample_at_eval=False,
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
            self.activation_inner = get_activation(
                activation_inner,
                session=self.session,
                training=self.training,
                from_logits=True,
                sample_at_train=sample_at_train,
                sample_at_eval=sample_at_eval
            )
            self.activation = get_activation(
                activation,
                session=self.session,
                training=self.training,
                from_logits=True,
                sample_at_train=sample_at_train,
                sample_at_eval=sample_at_eval
            )
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
            return_cell_state=False,
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
        self.return_cell_state = return_cell_state
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
                        return_state=self.return_cell_state,
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
                if self.return_cell_state:
                    H, _, c = H

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

                if self.return_cell_state:
                    return H, c

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

    def regularize(self, var, regularizer):
        if regularizer is not None:
            with self.session.as_default():
                with self.session.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)

    def get_regularizer_losses(self):
        return self.regularizer_losses[:]

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
            l2_normalize_states=False,
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
        self.l2_normalize_states = l2_normalize_states
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
                        l2_normalize_states=self.l2_normalize_states,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        epsilon=self.epsilon,
                    )

                    self.cell.build(inputs.shape[1:])

            self.built = True

    def __call__(self, inputs, mask=None, return_state=False):
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

                H, c = H

                if not self.return_sequences:
                    H = H[:, -1]
                    if return_state:
                        c = c[:, -1]

                if return_state:
                    out = (H, c)
                else:
                    out = H

                return out

class DropoutLayer(object):
    def __init__(
            self,
            rate,
            training=True,
            noise_shape=None,
            session=None,
     ):
        self.rate = rate
        self.training = training
        self.noise_shape = noise_shape
        self.training = training
        self.session = get_session(session)

    def __call__(self, inputs):
        shape = []
        tile_shape = []
        shape_src = tf.shape(inputs)

        for i, s in enumerate(self.noise_shape):
            if s is None:
                shape.append(shape_src[i])
            else:
                shape.append(s)
            tile_shape.append(shape_src[i] // shape[-1])

        def train_func():
            noise = tf.random_uniform(shape) > self.rate
            noise = tf.tile(
                noise,
                tile_shape
            )
            alt = tf.zeros_like(inputs)
            out = tf.where(noise, inputs, alt)
            return out

        def eval_func():
            return inputs

        return tf.cond(self.training, train_func, eval_func)


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


AttentionalLSTMDecoderStateTuple = collections.namedtuple(
    'AttentionalLSTMDecoderStateTuple',
    ' '.join(['h', 'c', 'a', 'y', 'mu'])
)


class AttentionalLSTMDecoderCell(LayerRNNCell):
    def __init__(
            self,
            num_hidden_units,
            num_output_units,
            keys=None,
            values=None,
            key_val_mask=None,
            gaussian_attn=False,
            attn_sigma2=0.25,
            training=False,
            num_query_units=None,
            project_keys=False,
            forget_bias=1.0,
            one_hot_inputs=False,
            sample_at_train=False,
            sample_at_eval=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            query_activation=None,
            key_activation=None,
            output_activation=None,
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            query_initializer='glorot_uniform_initializer',
            key_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            query_regularizer=None,
            key_regularizer=None,
            projection_regularizer=None,
            bias_regularizer=None,
            bottomup_dropout=None,
            recurrent_dropout=None,
            query_dropout=None,
            key_dropout=None,
            projection_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            implementation=1,
            l2_normalize_states=False,
            reuse=None,
            name=None,
            dtype=tf.float32,
            epsilon=1e-8,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                super(AttentionalLSTMDecoderCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                self.num_hidden_units = num_hidden_units
                self.num_output_units = num_output_units
                self.gaussian_attn = gaussian_attn
                self.attn_sigma2 = attn_sigma2
                self.training = training
                self.forget_bias = forget_bias
                self.one_hot_inputs = one_hot_inputs
                self.activation = get_activation(activation, training=self.training, session=self.session)
                self.recurrent_activation = get_activation(recurrent_activation, training=self.training, session=self.session)
                self.query_activation = get_activation(query_activation, training=self.training, session=self.session)
                self.key_activation = get_activation(
                    key_activation,
                    session=self.session,
                    training=self.training,
                    from_logits=True,
                    sample_at_train=sample_at_train,
                    sample_at_eval=sample_at_eval
                )
                self.output_activation = get_activation(
                    output_activation,
                    session=self.session,
                    training=self.training,
                    from_logits=True,
                    sample_at_train=sample_at_train,
                    sample_at_eval=sample_at_eval
                )
                self.bottomup_initializer = get_initializer(bottomup_initializer, session=self.session)
                self.recurrent_initializer = get_initializer(recurrent_initializer, session=self.session)
                self.query_initializer = get_initializer(query_initializer, session=self.session)
                self.key_initializer = get_initializer(key_initializer, session=self.session)
                self.bias_initializer = get_initializer(bias_initializer, session=self.session)
                self.bottomup_regularizer = get_regularizer(bottomup_regularizer, session=self.session)
                self.recurrent_regularizer = get_regularizer(recurrent_regularizer, session=self.session)
                self.query_regularizer = get_regularizer(query_regularizer, session=self.session)
                self.key_regularizer = get_regularizer(key_regularizer, session=self.session)
                self.projection_regularizer = get_regularizer(projection_regularizer, session=self.session)
                self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
                self.bottomup_dropout = get_dropout(bottomup_dropout, training=self.training, session=self.session)
                self.recurrent_dropout = get_dropout(recurrent_dropout, training=self.training, session=self.session)
                self.query_dropout = get_dropout(query_dropout, training=self.training, session=self.session)
                self.key_dropout = get_dropout(key_dropout, training=self.training, session=self.session)
                self.projection_dropout = get_dropout(projection_dropout, training=self.training, session=self.session)
                self.weight_normalization = weight_normalization
                self.layer_normalization = layer_normalization
                self.use_bias = use_bias
                self.implementation = implementation
                self.l2_normalize_states = l2_normalize_states
                assert self.implementation in [1,2], 'AttentionalLSTMDecoderCell implementation must be 1 or 2.'
                self.epsilon = epsilon

                self.keys = keys
                if self.keys is None:
                    self.num_attn_units = 0
                else:
                    self.num_attn_units = self.keys.shape[-2]
                    self.keys = self.key_activation(self.keys)

                self.values = values
                self.key_val_mask = key_val_mask
                if self.key_val_mask is not None:
                    self.key_val_mask_expanded = self.key_val_mask[..., None]
                else:
                    self.key_val_mask_expanded = None
                if num_query_units:
                    self.num_query_units = num_query_units
                elif self.keys is not None and self.keys.shape[-1] is not None:
                    self.num_query_units = self.keys.shape[-1]
                else:
                    self.num_query_units = None
                if project_keys or (self.keys is not None and self.keys.shape[-1] is not None and self.num_query_units != self.keys.shape[-1]):
                    self.project_keys = True
                else:
                    self.project_keys = False

                self.regularizer_map = {}
                self.regularization_initialized = False

    def add_regularization(self, var, regularizer):
        if regularizer is not None:
            with self.session.as_default():
                with self.session.graph.as_default():
                    self.regularizer_map[var] = regularizer

    def initialize_regularization(self):
        assert self.built, "Cannot initialize regularization before calling the layer because the weight matrices haven't been built."

        if not self.regularization_initialized:
            for var in tf.trainable_variables(scope=self.name):
                n1, n2 = var.name.split('/')[-2:]
                if 'bias' in n2:
                    self.add_regularization(var, self.bias_regularizer)
                if 'kernel' in n2:
                    if 'bottomup' in n1:
                        self.add_regularization(var, self.bottomup_regularizer)
                    elif 'recurrent' in n1:
                        self.add_regularization(var, self.recurrent_regularizer)
                    elif 'query' in n1:
                        self.add_regularization(var, self.query_regularizer)
                    elif 'key' in n1:
                        self.add_regularization(var, self.key_regularizer)
            self.regularization_initialized = True

    def get_regularization(self):
        self.initialize_regularization()
        return self.regularizer_map.copy()

    @property
    def state_size(self):
        return AttentionalLSTMDecoderStateTuple(
            h=self.num_hidden_units,
            c=self.num_hidden_units,
            a=self.num_attn_units,
            y=self.num_output_units,
            mu=1
        )

    @property
    def output_size(self):
        return self.state_size

    def norm(self, inputs, name):
        with self.session.as_default():
            with self.session.graph.as_default():
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
        with self.session.as_default():
            with self.session.graph.as_default():
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                     % inputs_shape)

                self.input_dims = inputs_shape[1].value

                bottomup_kernel = [
                    make_lambda(
                        DenseLayer(
                            training=self.training,
                            units=self.num_hidden_units * 4,
                            use_bias=False,
                            kernel_initializer=self.bottomup_initializer,
                            bias_initializer=self.bias_initializer,
                            activation=None,
                            normalize_weights=self.weight_normalization,
                            session=self.session,
                            reuse=tf.AUTO_REUSE,
                            name='bottomup'
                        ), session=self.session
                    ),
                    make_lambda(self.bottomup_dropout)
                ]
                self.bottomup_kernel = compose_lambdas(bottomup_kernel)

                if self.implementation == 1:
                    recurrent_kernel = [
                        make_lambda(
                            DenseLayer(
                                training=self.training,
                                units=self.num_hidden_units * 4,
                                use_bias=False,
                                kernel_initializer=self.recurrent_initializer,
                                bias_initializer=self.bias_initializer,
                                activation=None,
                                normalize_weights=self.weight_normalization,
                                session=self.session,
                                reuse=tf.AUTO_REUSE,
                                name='recurrent'
                            )
                        ),
                        make_lambda(self.recurrent_dropout)
                    ]
                    self.recurrent_kernel = compose_lambdas(recurrent_kernel)

                if not self.layer_normalization and self.use_bias:
                    self.bias = self.add_variable(
                        'bias',
                        shape=[1, self.num_hidden_units * 4],
                        initializer=self.bias_initializer
                    )
                else:
                    self.bias = 0

                projection = [
                    make_lambda(
                        DenseLayer(
                            training=self.training,
                            units=self.num_output_units,
                            use_bias=self.use_bias,
                            kernel_initializer=self.bottomup_initializer,
                            bias_initializer=self.bias_initializer,
                            activation=None,
                            normalize_weights=self.weight_normalization,
                            session=self.session,
                            reuse=tf.AUTO_REUSE,
                            name='projection'
                        )
                    ),
                    make_lambda(self.projection_dropout)
                ]
                self.projection = compose_lambdas(projection)

                if self.keys is not None or self.gaussian_attn:
                    if self.gaussian_attn:
                        q_units = 1
                    else:
                        q_units = self.num_query_units
                    q_kernel = [
                        make_lambda(
                            DenseLayer(
                                training=self.training,
                                units=q_units,
                                use_bias=self.use_bias,
                                kernel_initializer=self.query_initializer,
                                bias_initializer=self.bias_initializer,
                                activation=None,
                                normalize_weights=self.weight_normalization,
                                session=self.session,
                                reuse=tf.AUTO_REUSE,
                                name='query'
                            )
                        ),
                        make_lambda(self.query_dropout)
                    ]
                    self.q_kernel = compose_lambdas(q_kernel)
    
                    if self.project_keys and not self.gaussian_attn:
                        k_kernel = [
                            make_lambda(
                                DenseLayer(
                                    training=self.training,
                                    units=self.num_query_units,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.key_initializer,
                                    bias_initializer=self.bias_initializer,
                                    activation=None,
                                    normalize_weights=self.weight_normalization,
                                    session=self.session,
                                    reuse=tf.AUTO_REUSE,
                                    name='key'
                                )
                            ),
                            make_lambda(self.key_dropout)
                        ]
                        self.k_kernel = compose_lambdas(k_kernel)
                    else:
                        self.k_kernel = lambda x: x

        self.built = True

    def call(self, inputs, state):
        # Input features can be 0-dim, in which case they just define the sequence length and only predictions are used as input to the next timestep
        with self.session.as_default():
            with self.session.graph.as_default():
                if isinstance(inputs, list):
                    inputs, mask = inputs
                else:
                    mask = None

                units = self.num_hidden_units
                h_prev = state.h
                c_prev = state.c
                a_prev = state.a
                a_prev = state.a
                y_prev = state.y
                mu_prev = state.mu

                y_prev = self.output_activation(y_prev)

                if self.values is not None:
                    q = self.q_kernel(h_prev)
                    if self.gaussian_attn:
                        mu = mu_prev
                        sigma2 = self.attn_sigma2
                        ix = tf.cast(tf.range(self.values.shape[1]), dtype=tf.float32)[None, ...]
                        a = tf.exp(-(mu-ix)**2/ tf.maximum(sigma2, self.epsilon)) * self.key_val_mask
                        a /= tf.maximum(tf.reduce_sum(a, axis=1, keepdims=True), self.epsilon)
                        context_vector = tf.reduce_sum(self.values * a[..., None], axis=1)
                        mu += tf.abs(tf.tanh(q))
                    else:
                        q = self.q_kernel(h_prev)
                        k = self.k_kernel(self.keys)
                        v = self.values
                        mu = mu_prev

                        context_vector, a = scaled_dot_attn(
                            q,
                            k,
                            v,
                            mask=self.key_val_mask,
                            epsilon=self.epsilon,
                            session=self.session
                        )
                    x = tf.concat([inputs, y_prev, context_vector], axis=-1)

                else:
                    x = tf.concat([inputs, y_prev], axis=-1)
                    a = tf.zeros(shape=[tf.shape(inputs)[0], 0], dtype=self.dtype)
                    mu = mu_prev

                s = self.bottomup_kernel(x) + self.bias
                if self.implementation == 1:
                    s += self.recurrent_kernel(h_prev)

                # Forget gate
                f = s[:, :units]
                if self.layer_normalization:
                    f = self.norm(f, 'f_ln')
                f = self.recurrent_activation(f + self.forget_bias)

                # Input gate
                i = s[:, units:units * 2]
                if self.layer_normalization:
                    i = self.norm(i, 'i_ln')
                i = self.recurrent_activation(i)

                # Output gate
                o = s[:, units * 2:units * 3]
                if self.layer_normalization:
                    o = self.norm(o, 'o_ln')
                o = self.recurrent_activation(o)

                # Cell proposal
                g = s[:, units * 3:units * 4]
                if self.layer_normalization:
                    g = self.norm(g, 'g_ln')
                g = self.activation(g)

                c = f * c_prev + i * g
                h = o * self.activation(c)
                
                if self.l2_normalize_states:
                    h = tf.nn.l2_normalize(h, epsilon=self.epsilon, axis=-1)

                y = self.projection(h)

                if mask is not None:
                    c = tf.where(mask, c, c_prev)
                    h = tf.where(mask, h, h_prev)
                    a = tf.where(mask, a, a_prev)
                    y = tf.where(mask, y, y_prev)
                    mu = tf.where(mask, mu, mu_prev)

                state = AttentionalLSTMDecoderStateTuple(
                    h=h,
                    c=c,
                    a=a,
                    y=y,
                    mu=mu
                )

                return state, state


class AttentionalLSTMDecoderLayer(object):
    def __init__(
            self,
            num_hidden_units,
            num_output_units,
            keys=None,
            values=None,
            key_val_mask=None,
            gaussian_attn=False,
            attn_sigma2=0.25,
            initial_state=None,
            training=False,
            num_query_units=None,
            project_keys=False,
            forget_bias=1.0,
            one_hot_inputs=False,
            sample_at_train=False,
            sample_at_eval=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            query_activation=None,
            key_activation=None,
            output_activation=None,
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            query_initializer='glorot_uniform_initializer',
            key_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            query_regularizer=None,
            key_regularizer=None,
            projection_regularizer=None,
            bias_regularizer=None,
            bottomup_dropout=None,
            recurrent_dropout=None,
            query_dropout=None,
            key_dropout=None,
            projection_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            implementation=1,
            l2_normalize_states=False,
            reuse=None,
            name=None,
            dtype=tf.float32,
            epsilon=1e-8,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                super(AttentionalLSTMDecoderLayer, self).__init__()

                self.num_hidden_units = num_hidden_units
                self.num_output_units = num_output_units
                self.keys = keys
                self.values = values
                self.key_val_mask = key_val_mask
                self.gaussian_attn = gaussian_attn
                self.attn_sigma2 = attn_sigma2
                self.initial_state = initial_state
                self.num_query_units = num_query_units
                self.project_keys = project_keys
                self.training = training
                self.forget_bias = forget_bias
                self.one_hot_inputs = one_hot_inputs
                self.sample_at_train = sample_at_train
                self.sample_at_eval = sample_at_eval
                self.activation = activation
                self.recurrent_activation = recurrent_activation
                self.query_activation = query_activation
                self.key_activation = key_activation
                self.output_activation = output_activation
                self.bottomup_initializer = bottomup_initializer
                self.recurrent_initializer = recurrent_initializer
                self.query_initializer = query_initializer
                self.key_initializer = key_initializer
                self.bias_initializer = bias_initializer
                self.bottomup_regularizer = bottomup_regularizer
                self.recurrent_regularizer = recurrent_regularizer
                self.query_regularizer = query_regularizer
                self.key_regularizer = key_regularizer
                self.projection_regularizer = projection_regularizer
                self.bias_regularizer = bias_regularizer
                self.bottomup_dropout = bottomup_dropout
                self.recurrent_dropout = recurrent_dropout
                self.query_dropout = query_dropout
                self.key_dropout = key_dropout
                self.projection_dropout = projection_dropout
                self.weight_normalization = weight_normalization
                self.layer_normalization = layer_normalization
                self.use_bias = use_bias
                self.implementation = implementation
                self.l2_normalize_states = l2_normalize_states
                self.reuse = reuse
                self.name = name
                self.dtype = dtype
                self.epsilon = epsilon

                self.key_matrix = None
                self.built = False

    def get_regularizer_losses(self):
        if self.built:
            return self.cell.get_regularization()
        raise ValueError(
            'Attempted to get regularizer losses from an HMLSTMSegmenter that has not been called on data.')

    def build(self, inputs=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.cell = AttentionalLSTMDecoderCell(
                    num_hidden_units=self.num_hidden_units,
                    num_output_units=self.num_output_units,
                    keys=self.keys,
                    values=self.values,
                    key_val_mask=self.key_val_mask,
                    gaussian_attn=self.gaussian_attn,
                    attn_sigma2=self.attn_sigma2,
                    num_query_units=self.num_query_units,
                    project_keys=self.project_keys,
                    training=self.training,
                    forget_bias=self.forget_bias,
                    one_hot_inputs=self.one_hot_inputs,
                    sample_at_train=self.sample_at_train,
                    sample_at_eval=self.sample_at_eval,
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    query_activation=self.query_activation,
                    key_activation=self.key_activation,
                    output_activation=self.output_activation,
                    bottomup_initializer=self.bottomup_initializer,
                    recurrent_initializer=self.recurrent_initializer,
                    query_initializer=self.query_initializer,
                    key_initializer=self.key_initializer,
                    bias_initializer=self.bias_initializer,
                    bottomup_regularizer=self.bottomup_regularizer,
                    recurrent_regularizer=self.recurrent_regularizer,
                    query_regularizer=self.query_regularizer,
                    key_regularizer=self.key_regularizer,
                    projection_regularizer=self.projection_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    bottomup_dropout=self.bottomup_dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    query_dropout=self.query_dropout,
                    key_dropout=self.key_dropout,
                    projection_dropout=self.projection_dropout,
                    weight_normalization=self.weight_normalization,
                    layer_normalization=self.layer_normalization,
                    use_bias=self.use_bias,
                    implementation=self.implementation,
                    l2_normalize_states=self.l2_normalize_states,
                    reuse=self.reuse,
                    name=self.name,
                    dtype=self.dtype,
                    epsilon=self.epsilon
                )

                self.cell.build(inputs.shape[1:])

        self.built = True

    def __call__(self, inputs, mask=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                if not self.built:
                    self.build(inputs)

                if mask is not None:
                    inputs = [inputs, mask]

                output, _ = tf.nn.dynamic_rnn(
                    self.cell,
                    inputs,
                    sequence_length=None,
                    initial_state=self.initial_state,
                    swap_memory=True,
                    dtype=tf.float32
                )

                if self.keys is not None:
                    self.key_matrix = self.cell.key_activation(self.cell.keys)
                    self.value_matrix = self.cell.values

                return output

    def get_regularization(self):
        return self.cell.get_regularization()



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


def construct_positional_encoding(
        n_timesteps,
        n_units=32,
        n_batch=1,
        positional_encoding_type='periodic',
        positional_encoding_transform=None,
        positional_encoding_activation=None,
        inner_activation=None,
        batch_normalization_decay=None,
        conv_kernel_size=3,
        training=True,
        name=None,
        session=None,
        float_type='float32'
):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if isinstance(float_type, str):
                FLOAT_TF = getattr(tf, float_type)
            else:
                FLOAT_TF = float_type

            # Create a representation of time to supply to decoder
            positional_encoding = None

            if positional_encoding_type:
                # Create a trainable matrix of weights by timestep
                if positional_encoding_type.lower() == 'weights':
                    positional_encoding = tf.get_variable(
                        'decoder_positional_encoding_src_%s' % name,
                        shape=[n_timesteps, n_units],
                        initializer=tf.initializers.random_normal,
                    )
                    positional_encoding = tf.tile(positional_encoding, [n_batch, 1, 1])

                # Create a set of periodic functions with trainable phase and frequency
                elif positional_encoding_type.lower() in ['periodic', 'transformer']:
                    time = tf.cast(tf.range(1, n_timesteps + 1), dtype=FLOAT_TF)[..., None]
                    n = n_units // 2

                    if positional_encoding_type.lower() == 'periodic':
                        coef = tf.exp(tf.linspace(-2., 2., n))[None, ...]
                    elif positional_encoding_type.lower() == 'transformer':
                        log_timescale_increment = tf.log(10000.) / tf.maximum(tf.cast(n, dtype=FLOAT_TF) - 1, 1e-8)
                        coef = (tf.exp(np.arange(n) * -log_timescale_increment))[None, ...]

                    sin = tf.sin(time * coef)
                    cos = tf.cos(time * coef)

                    positional_encoding = tf.concat([sin, cos], axis=-1)

                else:
                    raise ValueError('Unrecognized decoder positional encoding type "%s".' % positional_encoding_type)

                # Transform the temporal encoding
                if positional_encoding_transform:
                    if name:
                        name_cur = name + '_positional_encoding_transform'
                    else:
                        name_cur = name

                    # RNN transform
                    if positional_encoding_transform.lower() == 'rnn':
                        positional_encoding = RNNLayer(
                            training=training,
                            units=n_units,
                            activation=tf.tanh,
                            recurrent_activation=tf.sigmoid,
                            return_sequences=True,
                            name=name_cur,
                            session=session
                        )(positional_encoding)

                    # CNN transform (1D)
                    elif positional_encoding_transform.lower() == 'cnn':
                        positional_encoding = Conv1DLayer(
                            conv_kernel_size,
                            training=training,
                            n_filters=n_units,
                            padding='same',
                            activation=inner_activation,
                            batch_normalization_decay=batch_normalization_decay,
                            name=name_cur,
                            session=session
                        )(positional_encoding)

                    # Dense transform
                    elif positional_encoding_transform.lower() == 'dense':
                        positional_encoding = DenseLayer(
                            training=training,
                            units=n_units,
                            activation=inner_activation,
                            batch_normalization_decay=batch_normalization_decay,
                            name=name_cur,
                            session=session
                        )(positional_encoding)

                    else:
                        raise ValueError(
                            'Unrecognized decoder temporal encoding transform "%s".' % positional_encoding_transform)

                # Apply activation function
                if positional_encoding_activation:
                    activation = get_activation(
                        positional_encoding_activation,
                        session=session,
                        training=training
                    )
                    positional_encoding = activation(positional_encoding)

            return positional_encoding


def preprocess_decoder_inputs(
        decoder_in,
        n_timesteps,
        units_decoder,
        training=True,
        decoder_hidden_state_expansion_type='tile',
        decoder_positional_encoding=None,
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

            # Create a representation of time to supply to decoder
            if decoder_positional_encoding is not None or decoder_positional_encoding_type:
                n_batch = tf.shape(decoder_in)[0]
                if decoder_positional_encoding is None:
                    if decoder_positional_encoding_transform or not decoder_positional_encoding_as_mask:
                        positional_encoding_units = decoder_positional_encoding_units
                    else:
                        positional_encoding_units = out.shape[-1]
                    decoder_positional_encoding = construct_positional_encoding(
                        n_timesteps,
                        positional_encoding_units,
                        n_batch,
                        positional_encoding_type=decoder_positional_encoding_type,
                        positional_encoding_transform=decoder_positional_encoding_transform,
                        positional_encoding_activation=decoder_positional_encoding_activation,
                        inner_activation=decoder_inner_activation,
                        batch_normalization_decay=decoder_batch_normalization_decay,
                        conv_kernel_size=decoder_conv_kernel_size,
                        training=training,
                        name=name,
                        session=session,
                        float_type=float_type
                    )[None, ...]
                    decoder_positional_encoding = tf.tile(decoder_positional_encoding, [n_batch, 1, 1])

                # Apply temporal encoding, either as mask or as extra features
                if decoder_positional_encoding_as_mask:
                    decoder_positional_encoding = tf.sigmoid(decoder_positional_encoding)
                    out = out * decoder_positional_encoding
                else:
                    out = tf.concat([out, decoder_positional_encoding], axis=-1)
            else:
                final_shape_positional_encoding = None

            return out, decoder_positional_encoding, flatten_batch, final_shape, final_shape_positional_encoding
