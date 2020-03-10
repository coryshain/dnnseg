import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import control_flow_ops, gradients, state_ops, resource_variable_ops, variables
from tensorflow.python.training.optimizer import _get_variable_for, _get_processor, Optimizer
from tensorflow.python.util import nest

from .backend import get_session


## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
def get_clipped_optimizer_class(base_optimizer_class, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            class ClippedOptimizer(base_optimizer_class):
                def __init__(self, *args, max_global_norm=None, **kwargs):
                    super(ClippedOptimizer, self).__init__(*args, **kwargs)
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
                    grads, _ = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)
                    vars = [v for _, v in grads_and_vars]
                    grads_and_vars = []
                    for grad, var in zip(grads, vars):
                        grads_and_vars.append((grad, var))

                    return super(ClippedOptimizer, self).apply_gradients(grads_and_vars, **kwargs)

            return ClippedOptimizer


def get_JTPS_optimizer_class(base_optimizer_class, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            class JTPSOptimizer(base_optimizer_class):
                def __init__(self, *args, meta_learning_rate=None, granularity='global', linking_fn='softplus', **kwargs):
                    super(JTPSOptimizer, self).__init__(*args, **kwargs)
                    if meta_learning_rate is None:
                        if len(args) > 0:
                            learning_rate = args[0]
                        else:
                            learning_rate = kwargs['learning_rate']
                        self._meta_learning_rate = learning_rate
                    else:
                        self._meta_learning_rate = meta_learning_rate
                    self.granularity = granularity.lower()
                    self.global_lambda = None
                    self.linking_fn = linking_fn
                    self._delta_prev = None
                    self._delta_prev_t = None
                    self._lambda = None
                    self._lambda_t = None
                    self._lambdas = None
                    self._previous_initialized = False
                    self._lambda_optimizer_class = self.__class__.__bases__[0]
                    if len(args) > 0:
                        args = list(args)
                        args[0] = self._meta_learning_rate
                    else:
                        kwargs['learning_rate'] = self._meta_learning_rate
                    self.lambda_optimizer = self._lambda_optimizer_class(*args, **kwargs)

                def get_linking_fn(self):
                    fn = self.linking_fn
                    if isinstance(fn, str):
                        if fn.lower() == 'identity':
                            out = tf.identity
                            inv = tf.identity
                        elif fn.lower() == 'softplus':
                            out = tf.nn.softplus
                            inv = tf.contrib.distributions.softplus_inverse
                        else:
                            raise ValueError('Unrecognized linking function "%s"' % fn)
                    else:
                        out, inv = fn

                    return out, inv

                def get_lambda(self, var):
                    if self.granularity == 'global':
                        return self.global_lambda
                    if self.granularity in ['cell', 'variable']:
                        return self.get_slot(var, 'lambda')

                    raise ValueError('Unrecognized value for parameter ``granularity``: "%s"' % self.granularity)

                def get_flattened_lambdas(self, var_list=None):
                    fn, _ = self.get_linking_fn()

                    if var_list is None:
                        var_list = tf.trainable_variables()

                    if self.granularity == 'global':
                        lambdas = self.global_lambda
                    elif self.granularity == 'variable':
                        lambdas = tf.stack(
                            [self.get_lambda(var) for var in var_list],
                            axis=0
                        )
                    else:
                        lambdas = tf.concat(
                            [tf.reshape(self.get_lambda(var), [-1]) for var in var_list],
                            axis=0
                        )

                    lambdas = fn(lambdas)

                    return lambdas

                def _create_slots(self, var_list):
                    _, fn_inv = self.get_linking_fn()

                    if self.granularity == 'global':
                        self.global_lambda = tf.Variable(
                            fn_inv(1.).eval(session=session),
                            trainable=False,
                            name='JTPS_lambda'
                        )

                    for v in var_list:
                        self._zeros_slot(v, 'delta', self._name)
                        self._zeros_slot(v, 'theta', self._name)
                        if self.granularity == 'cell':
                            self._get_or_make_slot_with_initializer(
                                v,
                                tf.constant_initializer(fn_inv(1.).eval(session=session)),
                                v.shape,
                                v.dtype,
                                "lambda",
                                self._name
                            )
                        elif self.granularity == 'variable':
                            self._get_or_make_slot_with_initializer(
                                v,
                                tf.constant_initializer(fn_inv(1.).eval(session=session)),
                                tf.TensorShape([]),
                                v.dtype,
                                "lambda",
                                self._name
                            )
                        elif self.granularity == 'global':
                            pass
                        else:
                            raise ValueError('Unrecognized value for parameter ``granularity``: "%s"' % self.granularity)

                    super(JTPSOptimizer, self)._create_slots(var_list)

                    if self.granularity == 'global':
                        lambdas = [self.global_lambda]
                    else:
                        lambdas = [self.get_slot(var, 'lambda') for var in var_list]
                    self.lambda_optimizer._create_slots(lambdas)
                    self.lambda_optimizer._prepare()

                def _apply_dense(self, grad, var):
                    theta_setter_op = self.get_slot(var, 'theta').assign(var)
                    with tf.control_dependencies([theta_setter_op]):
                        base_update_op = super(JTPSOptimizer, self)._apply_dense(grad, var)
                        return base_update_op

                def _apply_sparse(self, grad, var):
                    raise NotImplementedError("Sparse gradient updates are not supported.")

                def apply_gradients(self, grads_and_vars, global_step=None, name=None):
                    """Apply gradients to variables.

                    This is the second part of `minimize()`. It returns an `Operation` that
                    applies gradients.

                    Args:
                      grads_and_vars: List of (gradient, variable) pairs as returned by
                        `compute_gradients()`.
                      global_step: Optional `Variable` to increment by one after the
                        variables have been updated.
                      name: Optional name for the returned operation.  Default to the
                        name passed to the `Optimizer` constructor.

                    Returns:
                      An `Operation` that applies the specified gradients. If `global_step`
                      was not None, that operation also increments `global_step`.

                    Raises:
                      TypeError: If `grads_and_vars` is malformed.
                      ValueError: If none of the variables have gradients.
                    """
                    # This is a default implementation of apply_gradients() that can be shared
                    # by most optimizers.  It relies on the subclass implementing the following
                    # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

                    base_update_op = super(JTPSOptimizer, self).apply_gradients(
                        grads_and_vars,
                        global_step=global_step,
                        name=name
                    )

                    with tf.control_dependencies([base_update_op]):
                        fn, _ = self.get_linking_fn()
                        lambda_update_op = []

                        if self.granularity == 'global':
                            lambda_grad = 0
                            for g, v in grads_and_vars:
                                if g is not None:
                                    delta_prev = self.get_slot(v, 'delta')
                                    v_cur = self.get_slot(v, 'theta')
                                    lambda_v = self.get_lambda(v)

                                    var_fn = v_cur + fn(lambda_v) * delta_prev
                                    lambda_grad_cur = g * tf.gradients(var_fn, lambda_v)[0]
                                    lambda_grad += tf.reduce_sum(lambda_grad_cur)

                            lambda_update_op.append(
                                self.lambda_optimizer._apply_dense(lambda_grad, self.global_lambda)
                            )
                        else:
                            for g, v in grads_and_vars:
                                if g is not None:
                                    delta_prev = self.get_slot(v, 'delta')
                                    v_cur = self.get_slot(v, 'theta')
                                    lambda_v = self.get_lambda(v)

                                    var_fn = v_cur + fn(lambda_v) * delta_prev
                                    lambda_grad_cur = g * tf.gradients(var_fn, lambda_v)[0]
                                    if self.granularity == 'variable':
                                        lambda_grad_cur = tf.reduce_sum(lambda_grad_cur)
                                    else:
                                        assert self.granularity == 'cell', 'Unrecognized granularity type "%s"' % self.granularity
                                    lambda_update_op += [
                                        self.lambda_optimizer._apply_dense(lambda_grad_cur, lambda_v),
                                    ]

                        lambda_update_op = control_flow_ops.group(*lambda_update_op)
                        with tf.control_dependencies([lambda_update_op]):
                            update_op = []
                            for g, v in grads_and_vars:
                                if g is not None:
                                    v_cur = self.get_slot(v, 'theta')
                                    delta_prev = self.get_slot(v, 'delta')
                                    delta_cur = delta_prev.assign(v - v_cur)
                                    if self.granularity == 'global':
                                        lambda_cur = self.global_lambda
                                    else:
                                        lambda_cur = self.get_slot(v, 'lambda')

                                    v_new = v_cur + fn(lambda_cur) * delta_cur

                                    update_op += [
                                        state_ops.assign(v, v_new),

                                    ]

                            update_op = control_flow_ops.group(*update_op)

                            return update_op


            return JTPSOptimizer