# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-09-16 20:53:17
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-04 15:22:01

import theano
import theano.tensor as T

import lasagne
from lasagne import utils
from collections import OrderedDict
import numpy as np

def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to return the gradients for

    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `theano.grad(loss_or_grads, params)`.

    Raises
    ------
    ValueError
        If `loss_or_grads` is a list of a different length than `params`, or if
        any element of `params` is not a shared variable (while we could still
        compute its gradient, we can never update it and want to fail early).
    """
    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
  """Adam updates

  Adam updates implemented as in [1]_.

  Parameters
  ----------
  loss_or_grads : symbolic expression or list of expressions
      A scalar loss expression, or a list of gradient expressions
  params : list of shared variables
      The variables to generate update expressions for
  learning_rate : float
      Learning rate
  beta1 : float
      Exponential decay rate for the first moment estimates.
  beta2 : float
      Exponential decay rate for the second moment estimates.
  epsilon : float
      Constant for numerical stability.

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  The paper [1]_ includes an additional hyperparameter lambda. This is only
  needed to prove convergence of the algorithm and has no practical use
  (personal communication with the authors), it is therefore omitted here.

  References
  ----------
  .. [1] Kingma, Diederik, and Jimmy Ba (2014):
         Adam: A Method for Stochastic Optimization.
         arXiv preprint arXiv:1412.6980.
  """
  all_grads = get_or_compute_grads(loss_or_grads, params)
  updates = OrderedDict()
  P = []

  for param, g_t, lr in zip(params, all_grads, learning_rate):
    if lr.get_value() > 0:
      t_prev = theano.shared(utils.floatX(0.))
      t = t_prev + 1
      one = T.constant(1)
      a_t = lr * T.sqrt(one-beta2 ** t)/(one - beta1 ** t)

      value = param.get_value(borrow=True)
      m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
      v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)

      m_t = beta1 * m_prev + (one - beta1) * g_t
      v_t = beta2 * v_prev + (one-beta2) * g_t ** 2
      step = a_t * m_t / (T.sqrt(v_t) + epsilon)

      updates[m_prev] = m_t
      updates[v_prev] = v_t
      updates[param] = param - step
      updates[t_prev] = t

      P += [t_prev, m_prev, v_prev]

  return updates, P
