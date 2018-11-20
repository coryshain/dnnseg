import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers.utils import conv_output_length


if hasattr(rnn_cell_impl, 'LayerRNNCell'):
    LayerRNNCell = rnn_cell_impl.LayerRNNCell
else:
    LayerRNNCell = rnn_cell_impl._LayerRNNCell

def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess


def get_activation(activation, session=None, training=True, from_logits=True):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            hard_sigmoid = tf.keras.backend.hard_sigmoid

            if activation:
                if activation.lower() == 'hard_sigmoid':
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

                    out = lambda x: tf.cond(training, lambda: sample_fn(x), lambda: round_fn(x))

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
                out = getattr(tf, initializer)
            else:
                out = initializer

            if 'glorot' in initializer:
                out = out()

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

            return out


def round_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            with ops.name_scope("BinaryRound") as name:
                with session.graph.gradient_override_map({"Round": "Identity"}):
                    return tf.round(x, name=name)


def bernoulli_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            with ops.name_scope("BernoulliSample") as name:
                with session.graph.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):
                    return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)


@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]


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


class HMLSTMCell(LayerRNNCell):
    def __init__(
            self,
            num_units,
            num_layers,
            forget_bias=1.0,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            boundary_activation='hard_sigmoid',
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
            weight_normalization=False,
            layer_normalization=False,
            refeed_boundary=False,
            power=None,
            boundary_slope_annealing_rate=None,
            state_slope_annealing_rate=None,
            state_discretizer=None,
            global_step=None,
            implementation=1,
            reuse=None,
            name=None,
            dtype=None,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                super(HMLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                if not isinstance(num_units, list):
                    self._num_units = [num_units] * num_layers
                else:
                    self._num_units = num_units

                assert len(
                    self._num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                self._num_layers = num_layers

                self._forget_bias = forget_bias

                self._activation = get_activation(activation, session=self.session)
                self._inner_activation = get_activation(inner_activation, session=self.session)
                self._recurrent_activation = get_activation(recurrent_activation, session=self.session)
                self._boundary_activation = get_activation(boundary_activation, session=self.session)

                self._bottomup_initializer = get_initializer(bottomup_initializer, session=self.session)
                self._recurrent_initializer = get_initializer(recurrent_initializer, session=self.session)
                self._topdown_initializer = get_initializer(topdown_initializer, session=self.session)
                self._boundary_initializer = get_initializer(boundary_initializer, session=self.session)
                self._bias_initializer = get_initializer(bias_initializer, session=self.session)

                self._bottomup_regularizer = get_regularizer(bottomup_regularizer, session=self.session)
                self._recurrent_regularizer = get_regularizer(recurrent_regularizer, session=self.session)
                self._topdown_regularizer = get_regularizer(topdown_regularizer, session=self.session)
                self._boundary_regularizer = get_regularizer(boundary_regularizer, session=self.session)
                self._bias_regularizer = get_regularizer(bias_regularizer, session=self.session)

                self.regularizer_losses = []

                self._weight_normalization = weight_normalization
                self._layer_normalization = layer_normalization
                self._refeed_boundary = refeed_boundary
                self._power = power
                self._boundary_slope_annealing_rate = boundary_slope_annealing_rate
                self._state_slope_annealing_rate = state_slope_annealing_rate
                if state_discretizer == 'bsn_round':
                    self._state_discretizer = lambda x: round_straight_through(x, session=session)
                elif state_discretizer == 'bsn_bernoulli':
                    self._state_discretizer = lambda x: bernoulli_straight_through(x, session=session)
                else:
                    self._state_discretizer = None
                # self._state_discretizer = get_activation(state_discretizer, session=self.session)
                self.global_step = global_step
                self._implementation = implementation

                self._epsilon = 1e-8

    def _regularize(self, var, regularizer):
        if regularizer is not None:
            with self.session.as_default():
                with self.session.graph.as_default():
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
                size = (self._num_units[l], 1)
            else:
                size = (self._num_units[l],)

            out.append(size)

        out = tuple(out)

        return out

    def norm(self, inputs, name):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = tf.contrib.layers.layer_norm(
                    inputs,
                    reuse=tf.AUTO_REUSE,
                    scope=name
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

                self._kernel_bottomup = []
                self._kernel_recurrent = []
                self._kernel_topdown = []
                self._bias_boundary = []

                self._bias = []

                if self._implementation == 2:
                    self._kernel_boundary = []

                for l in range(self._num_layers):
                    if l == 0:
                        bottom_up_dim = inputs_shape[1].value
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
                        kernel_bottomup = kernel_bottomup / (
                                    tf.norm(kernel_bottomup, axis=0) + self._epsilon) * kernel_bottomup_g
                    self._regularize(kernel_bottomup, self._bottomup_regularizer)
                    self._kernel_bottomup.append(kernel_bottomup)

                    kernel_recurrent = self.add_variable(
                        'kernel_recurrent_%d' % l,
                        shape=[recurrent_dim, output_dim],
                        initializer=self._recurrent_initializer
                    )
                    if self._weight_normalization:
                        kernel_recurrent_g = tf.Variable(tf.ones([1, output_dim]), name='kernel_recurrent_g_%d' % l)
                        kernel_recurrent = kernel_recurrent / (
                                    tf.norm(kernel_recurrent, axis=0) + self._epsilon) * kernel_recurrent_g
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
                            kernel_topdown = kernel_topdown / (
                                        tf.norm(kernel_topdown, axis=0) + self._epsilon) * kernel_topdown_g
                        self._regularize(kernel_topdown, self._topdown_regularizer)
                        self._kernel_topdown.append(kernel_topdown)

                    bias = self.add_variable(
                        'bias_%d' % l,
                        shape=[1, output_dim],
                        initializer=self._bias_initializer
                    )
                    self._regularize(bias, self._bias_regularizer)
                    self._bias.append(bias)

                    if self._implementation == 1:
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
                            kernel_boundary = kernel_boundary / (
                                        tf.norm(kernel_boundary, axis=0) + self._epsilon) * kernel_boundary_g
                        self._regularize(kernel_boundary, self._boundary_regularizer)
                        self._kernel_boundary.append(kernel_boundary)

                        bias_boundary = self.add_variable(
                            'bias_boundary_%d' % l,
                            shape=[1, 1],
                            initializer=self._bias_initializer
                        )
                        self._regularize(bias_boundary, self._bias_regularizer)
                        self._bias_boundary.append(bias_boundary)

                if self._boundary_slope_annealing_rate and self.global_step is not None:
                    rate = self._boundary_slope_annealing_rate
                    # self.boundary_slope_coef = tf.sqrt(1 + rate * tf.cast(self.global_step, dtype=tf.float32))
                    self.boundary_slope_coef = tf.minimum(10., 1 + rate * tf.cast(self.global_step, dtype=tf.float32))

                if self._state_slope_annealing_rate and self.global_step is not None:
                    rate = self._state_slope_annealing_rate
                    # self.state_slope_coef = tf.sqrt(1 + rate * tf.cast(self.global_step, dtype=tf.float32))
                    self.state_slope_coef = tf.minimum(10., 1 + rate * tf.cast(self.global_step, dtype=tf.float32))

        self.built = True

    def call(self, inputs, state):
        with self.session.as_default():
            with self.session.graph.as_default():
                new_output = []
                new_state = []

                h_below = inputs
                z_below = None

                if self._power:
                    power = self._power
                else:
                    power = 1

                for l, layer in enumerate(state):
                    # Alias useful variables
                    if l < self._num_layers - 1:
                        # Use inner activation if non-final layer
                        activation = self._inner_activation
                    else:
                        # Use outer activation if final layer
                        activation = self._activation
                    units = self._num_units[l]

                    # EXTRACT DEPENDENCIES (c_behind, h_behind, h_below, h_above, z_behind, z_below):

                    # c_behind: Previous cell state at current layer
                    c_behind = layer[0]

                    # h_behind: Previous hidden state at current layer
                    h_behind = layer[1]

                    # h_above: hidden state of layer above at previous timestep (implicitly 0 if final layer)
                    if l < self._num_layers - 1:
                        h_above = state[l + 1][1]
                    else:
                        h_above = None

                    # z_behind: Previous boundary probability at current layer (implicitly 1 if final layer)
                    if l < self._num_layers - 1:
                        z_behind = layer[2]
                    else:
                        z_behind = None

                    # Bottom-up features
                    if self._implementation == 1 and self._refeed_boundary:
                        h_below = tf.concat([z_behind, h_below], axis=1)
                    s_bottomup = tf.matmul(h_below, self._kernel_bottomup[l])
                    if l > 0:
                        s_bottomup = s_bottomup * (z_below ** power)
                    if self._layer_normalization:
                        s_bottomup = self.norm(s_bottomup, 's_bottomup_ln_%d' % l)

                    # Recurrent features
                    s_recurrent = tf.matmul(h_behind, self._kernel_recurrent[l])
                    if self._layer_normalization:
                        s_recurrent = self.norm(s_recurrent, 's_recurrent_ln_%d' % l)

                    # Sum bottom-up and recurrent features
                    s = s_bottomup + s_recurrent

                    # Top-down features (if non-final layer)
                    if l < self._num_layers - 1:
                        # Compute top-down features
                        s_topdown = tf.matmul(h_above, self._kernel_topdown[l]) * (z_behind ** power)
                        if self._layer_normalization:
                            s_topdown = self.norm(s_topdown, 's_topdown_ln_%d' % l)
                        # Add in top-down features
                        s = s + s_topdown

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
                        if self._implementation == 2:
                            z_in = h
                            if self._refeed_boundary:
                                z_in = tf.concat([z_behind, z_in], axis=1)
                            # In implementation 2, boundary is a function of the hidden state
                            z = tf.matmul(z_in, self._kernel_boundary[l])
                        else:
                            # In implementation 1, boundary has its own slice of the hidden state preactivations
                            z = s[:, units * 4:]

                        z = z + self._bias_boundary[l]

                        if self._boundary_slope_annealing_rate and self.global_step is not None:
                            z *= self.boundary_slope_coef

                        z = self._boundary_activation(z)

                        # if l > 0:
                        #     z = z * z_below

                    if l < self._num_layers - 1:
                        output_l = (h, z)
                        new_state_l = (c, h, z)
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
            forget_bias=1.0,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            boundary_activation='hard_sigmoid',
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
            weight_normalization=False,
            layer_normalization=False,
            refeed_boundary=False,
            power=None,
            boundary_slope_annealing_rate=None,
            state_slope_annealing_rate=None,
            state_discretizer=None,
            global_step=None,
            implementation=1,
            reuse=None,
            name=None,
            dtype=None,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                if not isinstance(num_units, list):
                    self.num_units = [num_units] * num_layers
                else:
                    self.num_units = num_units

                assert len(
                    self.num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                self.num_layers = num_layers
                self.forget_bias = forget_bias

                self.activation = activation
                self.inner_activation = inner_activation
                self.recurrent_activation = recurrent_activation
                self.boundary_activation = boundary_activation

                self.bottomup_initializer = bottomup_initializer
                self.recurrent_initializer = recurrent_initializer
                self.topdown_initializer = topdown_initializer
                self.boundary_initializer = boundary_initializer
                self.bias_initializer = bias_initializer

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
                self.state_discretizer = state_discretizer
                self.global_step = global_step
                self.implementation = implementation

                self.reuse = reuse
                self.name = name
                self.dtype = dtype

                self.built = False

    def get_regularizer_losses(self):
        if self.built:
            return self.cell.get_regularizer_losses()
        raise ValueError(
            'Attempted to get regularizer losses from an HMLSTMSegmenter that has not been called on data.')

    def build(self, inputs=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.cell = HMLSTMCell(
                    self.num_units,
                    self.num_layers,
                    forget_bias=self.forget_bias,
                    activation=self.activation,
                    inner_activation=self.inner_activation,
                    recurrent_activation=self.recurrent_activation,
                    boundary_activation=self.boundary_activation,
                    bottomup_initializer=self.bottomup_initializer,
                    recurrent_initializer=self.recurrent_initializer,
                    topdown_initializer=self.topdown_initializer,
                    bias_initializer=self.bias_initializer,
                    bottomup_regularizer=self.bottomup_regularizer,
                    recurrent_regularizer=self.recurrent_regularizer,
                    topdown_regularizer=self.topdown_regularizer,
                    boundary_regularizer=self.boundary_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    weight_normalization=self.weight_normalization,
                    layer_normalization=self.layer_normalization,
                    refeed_boundary=self.refeed_boundary,
                    power=self.power,
                    boundary_slope_annealing_rate=self.boundary_slope_annealing_rate,
                    state_slope_annealing_rate=self.state_slope_annealing_rate,
                    state_discretizer=self.state_discretizer,
                    global_step=self.global_step,
                    implementation=self.implementation,
                    reuse=self.reuse,
                    name=self.name,
                    dtype=self.dtype,
                    session=self.session
                )

                self.cell.build(inputs.shape[1:])

        self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                if mask is not None:
                    sequence_length = tf.reduce_sum(mask, axis=1)
                else:
                    sequence_length = None
                output, state = tf.nn.dynamic_rnn(
                    self.cell,
                    inputs,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

                out = HMLSTMOutput(output)

                return out


class HMLSTMOutput(object):
    def __init__(self, output, session=None):
        self.session = get_session(session)

        self.num_layers = len(output)
        self.l = [HMLSTMOutputLevel(level, session=self.session) for level in output]

    def state(self, level=None, discrete=False, method='round'):
        if level is None:
            out = tuple([l.state(discrete=discrete, discretization_method=method) for l in self.l])
        else:
            out = self.l[level].state(discrete=discrete, discretization_method=method)

        return out

    def boundary(self, level=None, discrete=False, method='round', as_logits=False):
        if level is None:
            out = tuple(
                [l.boundary(discrete=discrete, discretization_method=method, as_logits=as_logits) for l in self.l[:-1]])
        else:
            out = self.l[level].boundary(discrete=discrete, discretization_method=method, as_logits=as_logits)

        return out

    def output(self, return_sequences=False):
        if return_sequences:
            out = self.state(level=-1, discrete=False)
        else:
            out = self.state(level=-1, discrete=False)[:, -1, :]

        return out


class HMLSTMOutputLevel(object):
    def __init__(self, output, session=None):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                self.h = output[0]
                if len(output) > 1:
                    self.z = output[1]
                else:
                    self.z = None

    def state(self, discrete=False, discretization_method='round'):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.h
                if discrete:
                    if discretization_method == 'round':
                        out = tf.cast(tf.round(self.h), dtype=tf.int32)
                    else:
                        raise ValueError('Discretization method "%s" not currently supported' % discretization_method)

                return out

    def boundary(self, discrete=False, discretization_method='round', as_logits=False):
        with self.session.as_default():
            with self.session.graph.as_default():
                if self.z is None:
                    out = tf.zeros(tf.shape(self.h)[:2])
                else:
                    out = self.z[..., 0]
                    if not discrete and as_logits:
                        out = sigmoid_to_logits(self.z[..., 0], session=self.session)

                if discrete:
                    if discretization_method == 'round':
                        out = tf.round(out)
                    else:
                        raise ValueError('Discretization method "%s" not currently supported' % discretization_method)

                if discrete:
                    out = tf.cast(out, dtype=tf.int32)

                return out


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
            refeed_discretized_outputs=False,
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

                self._refeed_discretized_outputs = refeed_discretized_outputs

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
                    if self._refeed_discretized_outputs and l == 0:
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

                # if self._refeed_outputs and l == 0 and self._num_layers > 1:
                if self._refeed_discretized_outputs:
                    prev_state_in = tf.argmax(state[-1][1], axis=-1)
                    prev_state_in = tf.one_hot(prev_state_in, state[-1][1].shape[-1])
                    layer_inputs.append(prev_state_in)

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
                        use_bias=self.use_bias
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
            batch_normalization_decay=0.9,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.activation = get_activation(activation, session=self.session, training=self.training)
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
            refeed_outputs = False,
            return_sequences=True,
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
        self.return_sequences = return_sequences
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
            refeed_discretized_outputs = False,
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
        self.refeed_discretized_outputs = refeed_discretized_outputs
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
                        refeed_discretized_outputs=self.refeed_discretized_outputs,
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
                else:
                    activation = activation

                rnn_layer = RNNLayer(
                    training=training,
                    units=units_encoder[l],
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    return_sequences=False,
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



