import sys
import os
import math
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers.utils import conv_output_length
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

from .data import get_random_permutation
from .kwargs import UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS
from .plot import plot_acoustic_features, plot_label_histogram, plot_label_heatmap, plot_binary_unit_heatmap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

if hasattr(rnn_cell_impl, 'LayerRNNCell'):
    LayerRNNCell = rnn_cell_impl.LayerRNNCell
else:
    LayerRNNCell = rnn_cell_impl._LayerRNNCell



def get_activation(activation):
    if activation:
        if activation == 'hard_sigmoid':
            out = tf.keras.backend.hard_sigmoid
        elif isinstance(activation, str):
            out = getattr(tf.nn, activation)
        else:
            out = activation
    else:
        out = lambda x: x

    return out


class MultiLSTMCell(LayerRNNCell):
    def __init__(
            self,
            num_units,
            num_layers,
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
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

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

                self._activation = get_activation(activation)
                self._inner_activation = get_activation(inner_activation)
                self._recurrent_activation = get_activation(recurrent_activation)

                self._kernel_initializer = self.get_initializer(kernel_initializer)
                self._bias_initializer = self.get_initializer(bias_initializer)

                self._refeed_outputs = refeed_outputs

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
                    layer_inputs.append(state[-1][1])

                # Compute gate pre-activations
                s = tf.matmul(
                    tf.concat(layer_inputs, axis=1),
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

    def get_initializer(self, initializer):
        with self.session.as_default():
            with self.session.graph.as_default():
                if isinstance(initializer, str):
                    out = getattr(tf, initializer)
                else:
                    out = initializer

                if 'glorot' in initializer:
                    out = out()

                return out


class SoftHMLSTMCell(LayerRNNCell):
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
            weight_normalization=False,
            layer_normalization=False,
            power=None,
            implementation=1,
            reuse=None,
            name=None,
            dtype=None,
            session=None
    ):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

        with self.session.as_default():
            with self.session.graph.as_default():
                super(SoftHMLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                if not isinstance(num_units, list):
                    self._num_units = [num_units] * num_layers
                else:
                    self._num_units = num_units

                assert len(self._num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                self._num_layers = num_layers
                self._forget_bias = forget_bias

                self._activation = get_activation(activation)
                self._inner_activation = get_activation(inner_activation)
                self._recurrent_activation = get_activation(recurrent_activation)
                self._boundary_activation = get_activation(boundary_activation)

                self._bottomup_initializer = self.get_initializer(bottomup_initializer)
                self._recurrent_initializer = self.get_initializer(recurrent_initializer)
                self._topdown_initializer = self.get_initializer(topdown_initializer)
                self._boundary_initializer = self.get_initializer(boundary_initializer)
                self._bias_initializer = self.get_initializer(bias_initializer)

                self.weight_normalization = weight_normalization
                self.layer_normalization = layer_normalization
                self.power = power
                self.implementation = implementation

                self.epsilon = 1e-8

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
                if not self.layer_normalization:
                    self._bias = []

                if self.implementation == 2:
                    self._kernel_boundary = []
                    if not self.layer_normalization:
                        self._bias_boundary = []

                for l in range(self._num_layers):
                    if l == 0:
                        bottom_up_dim = inputs_shape[1].value
                    else:
                        bottom_up_dim = self._num_units[l-1]

                    recurrent_dim = self._num_units[l]

                    if self.implementation == 1:
                        if l < self._num_layers - 1:
                            output_dim = 4 * self._num_units[l] + 1
                        else:
                            output_dim = 4 * self._num_units[l]
                    else:
                        output_dim = 4 * self._num_units[l]

                    kernel_bottomup = self.add_variable(
                        'kernel_bottomup_%d' %l,
                        shape=[bottom_up_dim, output_dim],
                        initializer=self._bottomup_initializer
                    )
                    self._kernel_bottomup.append(kernel_bottomup)

                    kernel_recurrent = self.add_variable(
                        'kernel_recurrent_%d' %l,
                        shape=[recurrent_dim, output_dim],
                        initializer=self._recurrent_initializer
                    )
                    self._kernel_recurrent.append(kernel_recurrent)

                    if l < self._num_layers - 1:
                        top_down_dim = self._num_units[l+1]
                        kernel_topdown = self.add_variable(
                            'kernel_topdown_%d' %l,
                            shape=[top_down_dim, output_dim],
                            initializer=self._topdown_initializer
                        )
                        self._kernel_topdown.append(kernel_topdown)

                    if not self.layer_normalization:
                        bias = self.add_variable(
                            'bias_%d' %l,
                            shape=[1, output_dim],
                            initializer=self._bias_initializer
                        )
                        self._bias.append(bias)

                    if self.implementation == 2:
                        kernel_boundary = self.add_variable(
                            'kernel_boundary_%d' %l,
                            shape=[self._num_units[l], 1],
                            initializer=self._boundary_initializer
                        )
                        self._kernel_boundary.append(kernel_boundary)

                        if not self.layer_normalization:
                            bias_boundary = self.add_variable(
                                'bias_boundary_%d' % l,
                                shape=[1, 1],
                                initializer=self._bias_initializer
                            )
                            self._bias_boundary.append(bias_boundary)
        self.built = True

    def call(self, inputs, state):
        with self.session.as_default():
            with self.session.graph.as_default():
                new_output = []
                new_state = []

                h_below = inputs
                z_below = None

                if self.power:
                    power = self.power
                else:
                    power = 1

                for l, layer in enumerate(state):
                    # EXTRACT DEPENDENCIES (c_behind, h_behind, h_below, h_above, z_behind, z_below):

                    # c_behind: Previous cell state at current layer
                    c_behind = layer[0]

                    # h_behind: Previous hidden state at current layer
                    h_behind = layer[1]

                    # h_below: Incoming features, either inputs (if first layer) or hidden state (if not first layer)
                    # z_below: Boundary probability of lower layer (implicitly 1 if first layer)
                    if l > 0:
                        h_below = new_output[-1][0]
                        z_below = self._boundary_activation(new_output[-1][1])

                    # h_above: hidden state of layer above at previous timestep (implicitly 0 if final layer)
                    if l < self._num_layers - 1:
                        h_above = state[l + 1][1]
                    else:
                        h_above = None

                    # z_behind: Previous boundary probability at current layer (implicitly 1 if final layer)
                    if l < self._num_layers - 1:
                        z_behind = self._boundary_activation(layer[2])
                    else:
                        z_behind = None

                    # Bottom-up features
                    s_bottomup = tf.matmul(h_below, self._kernel_bottomup[l])
                    if l > 0:
                        s_bottomup = s_bottomup * (z_below ** power)

                    # Recurrent features
                    s_recurrent = tf.matmul(h_behind, self._kernel_recurrent[l])

                    # Sum bottom-up and recurrent features
                    s = s_bottomup + s_recurrent

                    # Top-down features (if non-final layer)
                    if l < self._num_layers - 1:
                        # Compute top-down features
                        s_topdown = tf.matmul(h_above, self._kernel_topdown[l]) * (z_behind ** power)
                        # Add in top-down features
                        s = s + s_topdown

                    # Alias useful variables
                    if l < self._num_layers - 1:
                        # Use inner activation if non-final layer
                        activation = self._inner_activation
                    else:
                        # Use outer activation if final layer
                        activation = self._activation
                    units = self._num_units[l]

                    # Forget gate
                    f = s[:, :units]
                    if self.weight_normalization:
                        f_g = tf.Variable(tf.ones([1, units]), name='f_g_%d' %l)
                        f = f / (tf.norm(f, axis=0) + self.epsilon) * f_g
                    if self.layer_normalization:
                        f = self.norm(f, 'f_%d' %l)
                    else:
                        f = f + self._bias[l][:, units]
                    f = self._recurrent_activation(f + self._forget_bias)

                    # Input gate
                    i = s[:, units:units*2]
                    if self.weight_normalization:
                        i_g = tf.Variable(tf.ones([1, units]), name='i_g_%d' %l)
                        i = i / (tf.norm(i, axis=0) + self.epsilon) * i_g
                    if self.layer_normalization:
                        i = self.norm(i, 'i_%d' %l)
                    else:
                        i = i + self._bias[l][:, units:units*2]
                    i = self._recurrent_activation(i)

                    # Output gate
                    o = s[:, units*2:units*3]
                    if self.weight_normalization:
                        o_g = tf.Variable(tf.ones([1, units]), name='o_g_%d' %l)
                        o = o / (tf.norm(o, axis=0) + self.epsilon) * o_g
                    if self.layer_normalization:
                        o = self.norm(o, 'o_%d' %l)
                    else:
                        o = o + self._bias[l][:, units*2:units*3]
                    o = self._recurrent_activation(o)

                    # Cell proposal
                    g = s[:, units*3:units*4]
                    if self.weight_normalization:
                        g_g = tf.Variable(tf.ones([1, units]), name='g_g_%d' %l)
                        g = g / (tf.norm(g, axis=0) + self.epsilon) * g_g
                    if self.layer_normalization:
                        g = self.norm(g, 'g_%d' %l)
                    else:
                        g = g + self._bias[l][:, units*3:units*4]
                    g = activation(g)

                    # Cell state update (forget-gated previous cell plus input-gated cell proposal)
                    c = f * c_behind + i * g
                    if self.layer_normalization:
                        c = self.norm(c, 'c_%d' %l)

                    if l > 0:
                        # If non-initial, current and hidden cell states proportionally to the boundary probability z_below.
                        # If z_below is small, the cell state will mostly be copied from the previous timestep.
                        # If z_below is large, the cell state will mostly be updated based on the inputs.
                        c = z_below * c + (1 - z_below) * c_behind

                    if l < self._num_layers - 1:
                        # If non-final, delete cell state proportionally to boundary probability z_behind.
                        # If z_behind is low, the cell state will mostly be retained.
                        # If z_behind is high, the cell state will mostly be deleted (replaced with the gated cell-proposal vector)
                        c =  z_behind * i * g + (1 - z_behind) * c

                    # Compute the gated output
                    h = o * activation(c)

                    # Compute probability of a copy operation.
                    # Equal to the joint probability of not z_behind and not z_below
                    # (when both z_behind and z_below are zero, the state is completely copied forward).
                    if z_behind is None:
                        z_behind_tmp = 0
                    else:
                        z_behind_tmp = z_behind

                    if z_below is None:
                        z_below_tmp = 1
                    else:
                        z_below_tmp = z_below

                    copy_prob = (1 - z_behind_tmp) * (1 - z_below_tmp)

                    # Mix gated output with the previous hidden state proportionally to the copy probability
                    h = copy_prob * h_behind + (1 - copy_prob) * h

                    # Compute the current boundary probability
                    if l < self._num_layers - 1:
                        if self.implementation == 2:
                            # In implementation 2, boundary is a function of the hidden state
                            z = tf.matmul(h, self._kernel_boundary[l])
                            if self.weight_normalization:
                                z_g = tf.Variable(tf.ones([1, 1]), name='z_g_%d' % l)
                                z = z / (tf.norm(z, axis=0) + self.epsilon) * z_g
                            if self.layer_normalization:
                                z = self.norm(z, 'z_%d' %l)
                            else:
                                z = z + self._bias_boundary[l]
                        else:
                            # In implementation 1, boundary has its own slice of the kernels
                            z = s[:, units*4:]
                            if self.weight_normalization:
                                z_g = tf.Variable(tf.ones([1, 1]), name='z_g_%d' % l)
                                z = z / (tf.norm(z, axis=0) + self.epsilon) * z_g
                            if self.layer_normalization:
                                z = self.norm(z, 'z_%d' %l)
                            else:
                                z = z + self._bias[l][:, units*4:]

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
                    if l < self._num_layers - 1:
                        z_below = z
                    else:
                        z_below = None

                # Create output and state tuples
                new_output = tuple(new_output)
                new_state = tuple(new_state)

                return new_output, new_state

    def get_initializer(self, initializer):
        with self.session.as_default():
            with self.session.graph.as_default():
                if isinstance(initializer, str):
                    out = getattr(tf, initializer)
                else:
                    out = initializer

                if 'glorot' in initializer:
                    out = out()

                return out


class SoftHMLSTMSegmenter(object):
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
            weight_normalization=False,
            layer_normalization=False,
            power=None,
            implementation=1,
            reuse=None,
            name=None,
            dtype=None,
            session=None
    ):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

        with self.session.as_default():
            with self.session.graph.as_default():
                if not isinstance(num_units, list):
                    self.num_units = [num_units] * num_layers
                else:
                    self.num_units = num_units

                assert len(self.num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

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

                self.weight_normalization = weight_normalization
                self.layer_normalization = layer_normalization
                self.power = power
                self.implementation = implementation

                self.reuse = reuse
                self.name = name
                self.dtype = dtype

                self.built = False

    def build(self, inputs=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.cell = SoftHMLSTMCell(
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
                    weight_normalization=self.weight_normalization,
                    layer_normalization=self.layer_normalization,
                    power=self.power,
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
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

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
            out = tuple([l.boundary(discrete=discrete, discretization_method=method, as_logits=as_logits) for l in self.l[:-1]])
        else:
            out = self.l[level].boundary(discrete=discrete, discretization_method=method, as_logits=as_logits)

        return out

    def output(self, return_sequences=False):
        if return_sequences:
            out = self.state(level=-1, discrete=False)
        else:
            out = self.state(level=-1, discrete=False)[:,-1,:]

        return out


class HMLSTMOutputLevel(object):
    def __init__(self, output, session=None):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

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
                    if discrete or not as_logits:
                        out = tf.sigmoid(self.z[..., 0])

                if discrete:
                    if discretization_method == 'round':
                        out = tf.round(out)
                    else:
                        raise ValueError('Discretization method "%s" not currently supported' % discretization_method)

                if discrete:
                    out = tf.cast(out, dtype=tf.int32)

                return out




class DenseLayer(object):

    def __init__(
            self,
            training,
            units=None,
            use_bias=True,
            activation=None,
            batch_normalization_decay=0.9,
            normalize_weights=False,
            session=None
    ):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.activation = get_activation(activation)
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
            training,
            units=None,
            use_bias=True,
            layers_inner=3,
            activation_inner=None,
            activation=None,
            batch_normalization_decay=0.9,
            project_inputs=False,
            session=None
         ):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.layers_inner = layers_inner
        self.activation_inner = get_activation(activation_inner)
        self.activation = get_activation(activation)
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
            training,
            kernel_size,
            n_filters=None,
            stride=1,
            padding='valid',
            use_bias=True,
            activation=None,
            batch_normalization_decay=0.9,
            session=None
    ):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

        self.training = training
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.activation = get_activation(activation)
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

                    self.conv_1d_layer =  tf.keras.layers.Conv1D(
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
            training,
            kernel_size,
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
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

        self.training = training
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.layers_inner = layers_inner
        self.activation = get_activation(activation)
        self.activation_inner = get_activation(activation_inner)
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
            units=None,
            layers=1,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs = False,
            return_sequences=True,
            name=None,
            session=None
    ):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

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
                        activation=self.activation,
                        inner_activation=self.inner_activation,
                        recurrent_activation=self.recurrent_activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        refeed_outputs=self.refeed_outputs,
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

    def __init__(self, k, **kwargs):

        self.k = k
        for kwarg in AcousticEncoderDecoder._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

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
                self.encoder_type.lower() in ['cnn_softhmlstm' ,'softhmlstm'] and \
                self.resample_inputs == self.resample_outputs and \
                self.n_layers_encoder == self.n_layers_decoder and \
                self.units_encoder == self.units_decoder[::-1]:
            self.regularize_correspondences = True
        else:
            self.regularize_correspondences = False

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.entropy_regularizer_scale:
                    self.entropy_regularizer = self._binary_entropy_with_logits_regularizer(scale=self.entropy_regularizer_scale)
                else:
                    self.entropy_regularizer = None

                if self.segment_encoding_correspondence_regularizer_scale:
                    # self.segment_encoding_correspondence_regularizer = self._mse_regularizer(scale=self.segment_encoding_correspondence_regularizer_scale)
                    # self.segment_encoding_correspondence_regularizer = self._cross_entropy_regularizer(scale=self.segment_encoding_correspondence_regularizer_scale)
                    self.segment_encoding_correspondence_regularizer = tf.contrib.layers.l1_regularizer(scale=self.segment_encoding_correspondence_regularizer_scale)
                else:
                    self.segment_encoding_correspondence_regularizer = None

    def _pack_metadata(self):
        md = {
            'k': self.k,
        }
        for kwarg in AcousticEncoderDecoder._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.k = md.pop('k')
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
                self.X = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps_input, self.n_coef * (self.order + 1)), name='X')
                self.X_mask = tf.placeholder(self.FLOAT_TF, shape=(None, self.n_timesteps_input), name='X_mask')

                self.X_feat_mean = tf.reduce_sum(self.X, axis=-2) / tf.reduce_sum(self.X_mask, axis=-1, keepdims=True)
                self.X_time_mean = tf.reduce_mean(self.X, axis=-1)

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
                self.batch_len = tf.shape(self.X)[0]

                self.loss_summary = tf.placeholder(tf.float32, name='loss_summary')
                self.homogeneity = tf.placeholder(tf.float32, name='homogeneity')
                self.completeness = tf.placeholder(tf.float32, name='completeness')
                self.v_measure = tf.placeholder(tf.float32, name='v_measure')

                self.training_batch_norm = tf.placeholder(tf.bool, name='training_batch_norm')
                self.training_dropout = tf.placeholder(tf.bool, name='training_dropout')

    def _initialize_encoder(self):
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

                encoder = self.X
                if self.input_dropout_rate is not None:
                    encoder = tf.layers.dropout(
                        encoder,
                        self.input_dropout_rate,
                        noise_shape=[tf.shape(encoder)[0], tf.shape(encoder)[1], 1],
                        training=self.training_dropout
                    )

                units_utt = self.k
                if self.emb_dim:
                    units_utt += self.emb_dim

                if self.encoder_type.lower() in ['cnn_softhmlstm', 'softhmlstm', 'hmlstm']:
                    if self.encoder_type == 'cnn_softhmlstm':
                        encoder = Conv1DLayer(
                            self.training_batch_norm,
                            self.conv_kernel_size,
                            n_filters=self.n_coef * (self.order + 1),
                            padding='same',
                            activation=tf.nn.elu,
                            batch_normalization_decay=self.encoder_batch_normalization_decay,
                            session=self.sess
                        )(encoder)

                    self.segmenter = SoftHMLSTMSegmenter(
                        self.units_encoder + [units_utt],
                        self.n_layers_encoder,
                        activation=tf.tanh,
                        inner_activation=self.encoder_inner_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        boundary_activation=self.encoder_boundary_activation,
                        layer_normalization=self.encoder_layer_normalization,
                        power=self.boundary_power,
                        implementation=2
                    )(encoder, mask=mask)

                    self.segmentation_logits = self.segmenter.boundary(discrete=False, as_logits=True)
                    for l in self.segmentation_logits:
                        self._regularize(
                            l,
                            self.entropy_regularizer
                        )
                    self.segmentation_probs = tf.sigmoid(self.segmentation_logits)
                    self.state_probs = self.segmenter.state(discrete=False)

                    encoder = self.segmenter.output(return_sequences=False)

                    if encoding_batch_normalization_decay:
                        encoder = tf.contrib.layers.batch_norm(
                            encoder,
                            decay=encoding_batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training_batch_norm,
                            updates_collections=None
                        )

                elif self.encoder_type.lower() in ['rnn', 'cnn_rnn']:
                    if self.encoder_type == 'cnn_rnn':
                        encoder = Conv1DLayer(
                            self.training_batch_norm,
                            self.conv_kernel_size,
                            n_filters=self.n_coef * (self.order + 1),
                            padding='same',
                            activation=tf.nn.elu,
                            batch_normalization_decay=self.encoder_batch_normalization_decay,
                            session=self.sess
                        )(encoder)

                    encoder = RNNLayer(
                        units=self.units_encoder + [units_utt],
                        layers=self.n_layers_encoder,
                        activation=self.encoder_activation,
                        inner_activation=self.encoder_inner_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        refeed_outputs=False,
                        return_sequences=False,
                        name='RNNEncoder',
                        session=self.sess
                    )(encoder, mask=mask)


                elif self.encoder_type.lower() == 'cnn':
                    encoder = Conv1DLayer(
                        self.training_batch_norm,
                        self.conv_kernel_size,
                        n_filters=self.n_coef * (self.order + 1),
                        padding='same',
                        activation=tf.nn.elu,
                        batch_normalization_decay=self.encoder_batch_normalization_decay,
                        session=self.sess
                    )(encoder)

                    for i in range(self.n_layers_encoder - 1):
                        if i > 0 and self.encoder_resnet_n_layers_inner:
                            encoder = Conv1DResidualLayer(
                                self.training_batch_norm,
                                self.conv_kernel_size,
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
                                self.training_batch_norm,
                                self.conv_kernel_size,
                                n_filters=self.units_encoder[i],
                                padding='causal',
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)

                    encoder = DenseLayer(
                        self.training_batch_norm,
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
                                self.training_batch_norm,
                                units=self.n_timesteps_input * self.units_encoder[i],
                                layers_inner=self.encoder_resnet_n_layers_inner,
                                activation=self.encoder_inner_activation,
                                activation_inner=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)
                        else:
                            encoder = DenseLayer(
                                self.training_batch_norm,
                                units=self.n_timesteps_input * self.units_encoder[i],
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)

                    encoder = DenseLayer(
                        self.training_batch_norm,
                        units=units_utt,
                        activation=self.encoder_activation,
                        batch_normalization_decay=encoding_batch_normalization_decay,
                        session=self.sess
                    )(encoder)

                else:
                    raise ValueError('Encoder type "%s" is not currently supported' %self.encoder_type)

                self.encoder = encoder

    def _initialize_classifier(self):
        self.encoding = None
        raise NotImplementedError

    def _augment_encoding(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.classify_utterance:
                    if self.binary_classifier:
                        self.labels = self._binary2integer(tf.round(self.encoding))
                        self.label_probs = self._bernoulli2categorical(self.encoding)
                        self.encoding_entropy = self._binary_entropy_with_logits(self.encoding_logits)
                        self.encoding_entropy_mean = tf.reduce_mean(self.encoding_entropy)
                        self._regularize(self.encoding_logits, self.entropy_regularizer)
                    else:
                        if self.classify_utterance:
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
                decoder = self.decoder_in
                if self.mask_padding:
                    mask = self.y_mask
                else:
                    mask = None

                if self.decoder_type.lower() in ['rnn', 'cnn_rnn']:
                    decoder = tf.tile(
                        decoder[..., None, :],
                        [1, tf.shape(self.y)[1], 1]
                    ) * self.y_mask[...,None]
                    # index = tf.range(self.y.shape[1])[None, ..., None]
                    # index = tf.tile(
                    #     index,
                    #     [self.batch_len, 1, 1]
                    # )
                    # index = tf.cast(index, dtype=self.FLOAT_TF)
                    # decoder = tf.concat([decoder, index], axis=2)

                    decoder = RNNLayer(
                        units=self.units_decoder + [self.frame_dim],
                        layers=self.n_layers_decoder,
                        activation=self.decoder_inner_activation,
                        inner_activation=self.decoder_inner_activation,
                        recurrent_activation=self.decoder_recurrent_activation,
                        refeed_outputs=self.n_layers_decoder > 1,
                        return_sequences=True,
                        name='RNNDecoder',
                        session=self.sess
                    )(decoder, mask=mask)

                    print(decoder)

                    decoder = DenseLayer(
                        self.training_batch_norm,
                        units=self.frame_dim,
                        activation=self.decoder_activation,
                        batch_normalization_decay=self.decoder_batch_normalization_decay,
                        session=self.sess
                    )(decoder)

                elif self.decoder_type.lower() == 'cnn':
                    assert self.n_timesteps_output is not None, 'n_timesteps_output must be defined when decoder_type == "cnn"'

                    decoder = DenseLayer(
                        self.training_batch_norm,
                        self.n_timesteps_output * self.units_decoder[0],
                        activation=tf.nn.elu,
                        batch_normalization_decay=self.decoder_batch_normalization_decay,
                        session=self.sess
                    )(decoder)

                    decoder = tf.reshape(decoder, (self.batch_len, self.n_timesteps_output, self.units_decoder[0]))

                    for i in range(self.n_layers_decoder - 1):
                        if i > 0 and self.decoder_resnet_n_layers_inner:
                            decoder = Conv1DResidualLayer(
                                self.training_batch_norm,
                                self.conv_kernel_size,
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
                                self.training_batch_norm,
                                self.conv_kernel_size,
                                n_filters=self.units_decoder[i],
                                padding='same',
                                activation=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)

                        if self.regularize_correspondences:
                            states = self.state_probs[self.n_layers_encoder - i - 2]
                            if self.reverse_targets:
                                states = states[:, ::-1, :]

                            correspondences = states - decoder
                            self._regularize(
                                correspondences,
                                tf.contrib.layers.l2_regularizer(self.segment_encoding_correspondence_regularizer_scale)
                            )

                    decoder = DenseLayer(
                            self.training_batch_norm,
                            units=self.n_timesteps_output * self.frame_dim,
                            activation=self.decoder_inner_activation,
                            batch_normalization_decay=False,
                            session=self.sess
                    )(tf.layers.Flatten()(decoder))

                    decoder = tf.reshape(decoder, (self.batch_len, self.n_timesteps_output, self.frame_dim))

                elif self.decoder_type.lower() == 'dense':
                    assert self.n_timesteps_output is not None, 'n_timesteps_output must be defined when decoder_type == "dense"'

                    for i in range(self.n_layers_decoder - 1):
                        decoder = tf.layers.Flatten()(decoder)

                        if i > 0 and self.decoder_resnet_n_layers_inner:
                            if self.units_decoder[i] != self.units_decoder[i-1]:
                                project_inputs = True
                            else:
                                project_inputs = False

                            decoder = DenseResidualLayer(
                                self.training_batch_norm,
                                units=self.n_timesteps_output * self.units_decoder[i],
                                layers_inner=self.decoder_resnet_n_layers_inner,
                                activation=self.decoder_inner_activation,
                                activation_inner=self.decoder_inner_activation,
                                project_inputs=project_inputs,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)
                        else:
                            decoder = DenseLayer(
                                self.training_batch_norm,
                                units=self.n_timesteps_output * self.units_decoder[i],
                                activation=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)

                        decoder = tf.reshape(decoder, (self.batch_len, self.n_timesteps_output, self.units_decoder[i]))

                        if self.regularize_correspondences:
                            states = self.state_probs[self.n_layers_encoder - i - 2]
                            if self.reverse_targets:
                                states = states[:, ::-1, :]

                            correspondences = states - decoder
                            self._regularize(
                                correspondences,
                                tf.contrib.layers.l2_regularizer(self.segment_encoding_correspondence_regularizer_scale)
                            )

                    decoder = tf.layers.Flatten()(decoder)
                    decoder = DenseLayer(
                        self.training_batch_norm,
                        units=self.n_timesteps_output * self.frame_dim,
                        activation=self.decoder_activation,
                        batch_normalization_decay=None,
                        session=self.sess
                    )(decoder)

                    decoder = tf.reshape(decoder, (self.batch_len, self.n_timesteps_output, self.frame_dim))

                else:
                    raise ValueError('Decoder type "%s" is not currently supported' %self.decoder_type)

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

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                tf.summary.scalar('loss_summary', self.loss_summary, collections=['metrics'])
                if self.classify_utterance:
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
                targets = tf.expand_dims(targets, axis=2)
                preds = tf.expand_dims(preds, axis=1)
                offsets = targets - preds

                distances = tf.norm(offsets, axis=3)

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
                b = tf.shape(D_i)[1]

                # Extract alignment scores from adjacent cells in preceding row, prepending upper-left alignment to R_im1_jm1
                R_im1_jm1 = tf.concat(
                    [
                        tf.fill([1,b], R_i0),
                        R_im1[:-1, :]
                    ],
                    axis=0
                )
                R_im1_j = R_im1

                # Scan over columns of D (prediction indices)
                out = tf.scan(
                    lambda a, x: self._dtw_compute_cell(x[0], x[1], x[2], a, gamma),
                    [D_i, R_im1_jm1, R_im1_j],
                    initializer=tf.fill([b], np.inf),
                    swap_memory=True
                )

                return out

    def _dtw_outer_scan(self, D, gamma):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Extract dimensions
                n = tf.shape(D)[0]
                m = tf.shape(D)[1]
                b = tf.shape(D)[2]

                # Construct the 0th column with appropriate dimensionality
                R_i0 = tf.concat([[0.], tf.fill([n-1], np.inf)], axis=0)

                # Scan over rows of D (target indices)
                out = tf.scan(
                    lambda a, x: self._dtw_inner_scan(x[0], a, x[1], gamma),
                    [D, R_i0],
                    initializer=tf.fill([m, b], np.inf),
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
                n = tf.shape(targets)[1]
                b = tf.shape(targets)[0]

                # Compute/transform mask as needed
                if mask is None:
                    mask = tf.ones([n, b])
                else:
                    mask = tf.transpose(mask, perm=[1, 0])

                # Compute distance matrix
                D = self._pairwise_distances(targets, preds)

                # Move batch dimension to end so we can scan along time dimensions
                D = tf.transpose(D, perm=[1, 2, 0])

                # Perform soft-DTW alignment
                R = self._dtw_outer_scan(D, gamma)

                # Move batch dimension back to beginning so indexing works as expected
                R = tf.transpose(R, perm=[2, 0, 1])

                # Extract final cell of alignment matrix
                if self.pad_seqs:
                    target_end_ix = tf.cast(
                        tf.maximum(
                            tf.reduce_sum(mask, axis=0) - 1,
                            tf.zeros([b], dtype=self.FLOAT_TF)
                        ),
                        self.INT_TF
                    )
                    out = tf.gather(R[:, :, -1], target_end_ix, axis=1)
                else:
                    out = R[:, -1, -1]

                # Take the batch mean
                out = tf.reduce_mean(out)

                self.R = R


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

    def _binary_entropy_with_logits(self, p):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = tf.minimum(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=p, labels=tf.zeros_like(p)),
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=p, labels=tf.ones_like(p))
                )

                return out

    def _binary_entropy_with_logits_regularizer(self, scale):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return lambda bit_probs: tf.reduce_mean(self._binary_entropy_with_logits(bit_probs)) * scale

    def _cross_entropy_regularizer(self, scale):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return lambda x: tf.nn.sigmoid_cross_entropy_with_logits(labels=x[0], logits=x[1]) * scale

    def _mse_regularizer(self, scale):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return lambda offsets: tf.reduce_mean(offsets ** 2) * scale

    def _regularize(self, var, regularizer):
        if regularizer is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)




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
            training_batch_norm=False,
            training_dropout=False,
            shuffle=False,
            plot=True,
            verbose=True
    ):
        binary = self.binary_classifier
        seg = self.encoder_type.lower() in ['cnn_softhmlstm', 'softhmlstm']

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

                if self.classify_utterance:
                    if verbose:
                        sys.stderr.write('Predicting labels...\n\n')

                    to_run = []

                    labels_pred = []
                    to_run.append(self.labels_post)

                    if binary:
                        encoding = []
                        encoding_entropy = []
                        to_run += [self.encoding_post, self.encoding_entropy]

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
                                indices = np.arange(i,i+minibatch_size)
                            else:
                                indices = i

                        fd_minibatch = {
                            self.X: X_cv[indices],
                            self.X_mask: X_mask_cv[indices],
                            self.y: y_cv[indices],
                            self.y_mask: y_mask_cv[indices],
                            self.training_batch_norm: training_batch_norm,
                            self.training_dropout: training_dropout
                        }

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

                    if shuffle:
                        labels_pred = labels_pred[perm_inv]

                    if binary:
                        encoding = np.concatenate(encoding, axis=0)
                        encoding_entropy = np.concatenate(encoding_entropy, axis=0).mean()
                        if shuffle:
                            encoding = encoding[perm_inv]

                    homogeneity = homogeneity_score(labels_cv, labels_pred)
                    completeness = completeness_score(labels_cv, labels_pred)
                    v_measure = v_measure_score(labels_cv, labels_pred)

                    if verbose:
                        if self.binary_classifier:
                            sys.stderr.write('Encoding entropy: %s\n\n' % encoding_entropy)
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

                else:
                    homogeneity = completeness = v_measure = 0.

                if plot:
                    if verbose:
                        sys.stderr.write('Plotting...\n\n')

                    if self.classify_utterance and ix2label is not None:
                        labels_string = np.vectorize(lambda x: ix2label[x])(labels_cv.astype('int'))
                        titles = labels_string[self.plot_ix]

                        plot_label_heatmap(
                            labels_string,
                            labels_pred.astype('int'),
                            dir=self.outdir
                        )
                        if binary:
                            plot_binary_unit_heatmap(
                                labels_string,
                                encoding,
                                dir=self.outdir
                            )

                    else:
                        titles = [None] * n_plot

                    to_run = [self.reconst]
                    if seg:
                        to_run += [self.segmentation_probs, self.state_probs]

                    if self.pad_seqs:
                        X_cv_plot = X_cv[self.plot_ix]
                        y_cv_plot = (y_cv[self.plot_ix] * y_mask_cv[self.plot_ix][..., None]) if self.normalize_data else y_cv[self.plot_ix]

                        fd_minibatch = {
                            self.X: X_cv[self.plot_ix] if self.pad_seqs else [X_cv[ix] for ix in self.plot_ix],
                            self.X_mask: X_mask_cv[self.plot_ix] if self.pad_seqs else [X_mask_cv[ix] for ix in self.plot_ix],
                            self.y: y_cv[self.plot_ix] if self.pad_seqs else [y_cv[ix] for ix in self.plot_ix],
                            self.y_mask: y_mask_cv[self.plot_ix] if self.pad_seqs else [y_mask_cv[ix] for ix in self.plot_ix],
                            self.training_batch_norm: training_dropout,
                            self.training_dropout: training_dropout
                        }

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
                                self.training_batch_norm: training_dropout,
                                self.training_dropout: training_dropout
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

                    self.plot_reconstructions(
                        X_cv_plot,
                        y_cv_plot,
                        reconst,
                        titles=titles,
                        target_means=y_means_cv[self.plot_ix] if self.residual_decoder else None,
                        segmentation_probs=segmentation_probs,
                        states=states,
                        drop_zeros=self.pad_seqs
                    )

                    if self.classify_utterance:
                        self.plot_label_histogram(labels_pred)

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
            n_plot=10,
            verbose=True
    ):
        if verbose:
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
            y_means_cv = None

        if n_iter is None:
            n_iter = self.n_iter

        if verbose:
            sys.stderr.write('*' * 100 + '\n')
            sys.stderr.write(self.report_settings())
            sys.stderr.write('*' * 100 + '\n\n')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.pad_seqs:
                    if not np.isfinite(self.minibatch_size):
                        minibatch_size = len(y)
                    else:
                        minibatch_size = self.minibatch_size
                    n_minibatch = math.ceil(float(len(y)) / minibatch_size)
                else:
                    minibatch_size = 1
                    n_minibatch = len(y)

                homogeneity, completeness, v_measure = self.run_incremental_evaluation(
                    X_cv,
                    X_mask_cv,
                    y_cv,
                    y_mask_cv,
                    labels_cv,
                    ix2label=ix2label,
                    y_means_cv=None if y_means_cv is None else y_means_cv,
                    plot=True,
                    verbose=verbose
                )

                while self.global_step.eval(session=self.sess) < n_iter:
                    perm, perm_inv = get_random_permutation(len(y))

                    if verbose:
                        t0_iter = time.time()
                        sys.stderr.write('-' * 50 + '\n')
                        sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        sys.stderr.write('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                        sys.stderr.write('Updating...\n')
                        pb = tf.contrib.keras.utils.Progbar(n_minibatch)
                    loss_total = 0.

                    for i in range(0, len(y), minibatch_size):
                        if self.pad_seqs:
                            indices = perm[i:i+minibatch_size]
                        else:
                            indices = perm[i]

                        fd_minibatch = {
                            self.X: X[indices],
                            self.X_mask: X_mask[indices],
                            self.y: y[indices],
                            self.y_mask: y_mask[indices],
                            self.training_batch_norm: True,
                            self.training_dropout: True
                        }

                        info_dict = self.run_train_step(fd_minibatch)
                        metric_cur = info_dict['loss']

                        if self.ema_decay:
                            self.sess.run(self.ema_op)
                        if not np.isfinite(metric_cur):
                            metric_cur = 0
                        loss_total += metric_cur

                        self.sess.run(self.incr_global_batch_step)
                        if verbose:
                            pb.update((i/minibatch_size)+1, values=[('loss', metric_cur)])

                        self.check_numerics()

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

                            homogeneity, completeness, v_measure = self.run_incremental_evaluation(
                                X_cv,
                                X_mask_cv,
                                y_cv,
                                y_mask_cv,
                                labels_cv,
                                ix2label=ix2label,
                                y_means_cv=None if y_means_cv is None else y_means_cv,
                                plot=True,
                                verbose=verbose
                            )

                            fd_summary = {
                                self.loss_summary: loss_total
                            }

                            if self.classify_utterance:
                                fd_summary[self.homogeneity] = homogeneity
                                fd_summary[self.completeness] = completeness
                                fd_summary[self.v_measure] = v_measure

                            summary_metrics = self.sess.run(self.summary_metrics, feed_dict=fd_summary)
                            self.writer.add_summary(summary_metrics, self.global_step.eval(session=self.sess))

                        else:
                            if verbose:
                                sys.stderr.write('Numerics check failed. Aborting save and reloading from previous checkpoint...\n')

                            self.load()

                    if verbose:
                        t1_iter = time.time()
                        sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

    def plot_reconstructions(
            self,
            inputs,
            targets,
            preds,
            drop_zeros=True,
            titles=None,
            segmentation_probs=None,
            states=None,
            target_means=None,
            dir=None
    ):
        if dir is None:
            dir = self.outdir

        plot_acoustic_features(
            inputs,
            targets,
            preds,
            drop_zeros=drop_zeros,
            titles=titles,
            segmentation_probs=segmentation_probs,
            states=states,
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
                            if self.ema_decay:
                                self.ema_saver.restore(self.sess, dir + '/model.ckpt')
                            else:
                                sys.stderr.write('EMA is not turned on for this model. Ignoring command to load into predict mode.\n')
                        else:
                            self.saver.restore(self.sess, dir + '/model.ckpt')
                    except:
                        if predict:
                            if self.ema_decay:
                                self.ema_saver.restore(self.sess, dir + '/model_backup.ckpt')
                            else:
                                sys.stderr.write('EMA is not turned on for this model. Ignoring command to load into predict mode.\n')
                        else:
                            self.saver.restore(self.sess, dir + '/model_backup.ckpt')
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')
                    self.sess.run(tf.global_variables_initializer())

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

    def __init__(self, k, **kwargs):
        super(AcousticEncoderDecoderMLE, self).__init__(
            k=k,
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

    def _initialize_classifier(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.encoding_logits = self.encoder[:, :self.k]

                if self.classify_utterance:
                    if self.binary_classifier:
                        self.encoding = tf.sigmoid(self.encoding_logits)
                    else:
                        self.encoding = tf.nn.softmax(self.encoding_logits)
                else:
                    self.encoding = tf.nn.elu(self.encoding_logits)

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
                if not self.dtw_gamma:
                    self.reconst *= self.y_mask[..., None]

                self.encoding_post = self.encoding
                if self.classify_utterance:
                    self.labels_post = self.labels
                    self.label_probs_post = self.label_probs

                if self.use_dtw:
                    loss = self._soft_dtw_B(self.y, self.out, self.dtw_gamma, mask=self.y_mask)
                else:
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

                if len(self.regularizer_losses) > 0:
                    self.regularizer_loss_total = tf.add_n(self.regularizer_losses)
                    loss = loss + self.regularizer_loss_total

                self.loss = loss
                self.optim = self._initialize_optimizer(self.optim_name)
                self.train_op = self.optim.minimize(self.loss, global_step=self.global_batch_step)

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
            if return_encoding_entropy:
                to_run.append(self.encoding_entropy_mean)
                to_run_names.append('encoding_entropy')
            if self.encoder_type.lower() in ['cnn_softhmlstm', 'softhmlstm'] and return_segmentation_probs:
                to_run.append(self.segmentation_probs)
                to_run_names.append('segmentation_probs')

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





