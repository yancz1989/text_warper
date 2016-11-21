# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-26 20:35:17
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-04 16:14:16

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import as_tuple
from tools import generateT
import cv2
from collections import OrderedDict

def const(x):
  return T.constant(np.array(x, dtype = theano.config.floatX))

def vector(var, l):
  return T.tile(var, l)

def constv(var, l, dtype = 'float32'):
  return T.tile(T.constant(var, dtype = dtype), l)

def transform_affine(para, input, method, dtype = 'float32', scale_factor = 1):
  '''
    scale_factor: 
      if equals 1.0, output the same size with the input image;
      if equals 2.0, output the double size with input image based on para and method
      if equals 0.5, output the halves size may lose contents.
      Usage: if you need to resize, you can set paras to 0.5 and scale_factor to 0.5
  '''
  num_batch, num_channels, height, width = input.shape
  para = T.reshape(para, (-1, 3, 3))

  # grid of (x_t, y_t, 1), eq (1) in ref [1], in scales, e.g. [-1, 1]
  out_height = T.cast(height / scale_factor[0], 'int64')
  out_width = T.cast(width / scale_factor[1], 'int64')
  grid = _meshgrid(out_height, out_width)

  # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
  T_g = T.dot(para, grid)
  x_s = T_g[:, 0] / (T_g[:, 2])
  y_s = T_g[:, 1] / (T_g[:, 2])
  x_s_flat = x_s.flatten()
  y_s_flat = y_s.flatten()

  # dimshuffle input to (bs, height, width, channels)
  input_dim = input.dimshuffle(0, 2, 3, 1)
  input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_height, out_width, dtype = dtype)

  output = T.reshape(input_transformed, (num_batch, out_height, out_width, num_channels))
  output = T.cast(output.dimshuffle(0, 3, 1, 2), 'float32')  # dimshuffle to conv format
  return output

def _interpolate(im, x, y, out_height, out_width, dtype = 'float32'):
  # *_f are floats
  num_batch, height, width, channels = im.shape
  height_f = T.cast(height, dtype = dtype)
  width_f = T.cast(width, dtype = dtype)

  # scale coordinates from [-1, 1] to [0, width/height - 1]
  idx = ((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)).nonzero()[0]
  # x = (x + 1) / 2 * (width_f - 1)
  # y = (y + 1) / 2 * (height_f - 1)
  x = x * (width_f - 1)
  y = y * (height_f - 1)
  # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
  # we need those in floatX for interpolation and in int64 for indexing. for
  # indexing, we need to take care they do not extend past the image.
  x0_f = T.floor(x)
  y0_f = T.floor(y)
  x1_f = x0_f + 1
  y1_f = y0_f + 1
  x0 = T.cast(x0_f, 'int64')
  y0 = T.cast(y0_f, 'int64')
  x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
  y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

  # The input is [num_batch, height, width, channels]. We do the lookup in
  # the flattened input, i.e [num_batch*height*width, channels]. We need
  # to offset all indices to match the flat version
  dim2 = width
  dim1 = width*height
  base = T.repeat(
      T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
  base_y0 = base + y0*dim2
  base_y1 = base + y1*dim2
  idx_a = base_y0 + x0
  idx_b = base_y1 + x0
  idx_c = base_y0 + x1
  idx_d = base_y1 + x1

  # use indices to lookup pixels for all samples
  im_flat = im.reshape((-1, channels))
  Ia = im_flat[idx_a[idx]]
  Ib = im_flat[idx_b[idx]]
  Ic = im_flat[idx_c[idx]]
  Id = im_flat[idx_d[idx]]

  # calculate interpolated values
  wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')[idx, :]
  wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')[idx, :]
  wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')[idx, :]
  wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')[idx, :]
  output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)

  # out = T.zeros_like(((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x'))
  out = T.zeros_like(im_flat)
  return T.set_subtensor(out[idx, :], output)

def _linspace(start, stop, num):
  # Theano linspace. Behaves similar to np.linspace
  start = T.cast(start, theano.config.floatX)
  stop = T.cast(stop, theano.config.floatX)
  num = T.cast(num, theano.config.floatX)
  step = (stop-start)/(num-1)
  return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
  # This function is the grid generator from eq. (1) in reference [1].
  # It is equivalent to the following numpy code:
  #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
  #                         np.linspace(-1, 1, height))
  #  ones = np.ones(np.prod(x_t.shape))
  #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
  # It is implemented in Theano instead to support symbolic grid sizes.
  # Note: If the image size is known at layer construction time, we could
  # compute the meshgrid offline in numpy instead of doing it dynamically
  # in Theano. However, it hardly affected performance when we tried.
  x_t = T.dot(T.ones((height, 1)),
              _linspace(0.0, 1.0, width).dimshuffle('x', 0))
  y_t = T.dot(_linspace(0.0, 1.0, height).dimshuffle(0, 'x'),
              T.ones((1, width)))

  x_t_flat = x_t.reshape((1, -1))
  y_t_flat = y_t.reshape((1, -1))
  ones = T.ones_like(x_t_flat)
  grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
  return grid

class PerspectiveLayer(lasagne.layers.base.MergeLayer):
  def __init__(self, incoming, localization_network, method,
      scale_factor=1, resize_factor = 1, dtype = 'float32', **kwargs):
    super(PerspectiveLayer, self).__init__(
        [incoming, localization_network], **kwargs)
    self.scale_factor = as_tuple(scale_factor, 2)
    self.dtype = dtype
    self.resize_factor = as_tuple(resize_factor, 2)
    input_shp, loc_shp = self.input_shapes
    self.method = method
    if len(input_shp) != 4:
        raise ValueError("The input network must have a 4-dimensional "
                         "output shape: (batch_size, num_input_channels, "
                         "input_rows, input_columns)")

  def get_output_shape_for(self, input_shapes):
    shape = input_shapes[0]
    factors = self.scale_factor
    return (shape[:2] + tuple(None if s is None else int(s / f)
                              for s, f in zip(shape[2:], factors)))

  def get_output_for(self, inputs, **kwargs):
    # see eq. (1) and sec 3.1 in [1]
    input, para = inputs
    num_batch, channels, height, width = input.shape
    _w = T.cast(width, dtype = self.dtype)
    _h = T.cast(height, dtype = self.dtype)
    mat = T.zeros((num_batch, 3, 3), dtype = self.dtype)
    mat = T.set_subtensor(mat[:, 0, 0], const(1.0))
    mat = T.set_subtensor(mat[:, 1, 1], const(1.0))
    mat = T.set_subtensor(mat[:, 2, 2], const(1.0))

    if self.method == 'perspective':
      mat = T.set_subtensor(mat[:, 2, 0], (para[:, 0] / 1e4 - 1e-3) * _w)
      mat = T.set_subtensor(mat[:, 2, 1], (para[:, 1] / 1e4 - 1e-3) * _h)
    elif self.method == 'angle':
      angle = T.cast(T.argmax(para, axis = 1), dtype = self.dtype) * np.pi / 90 - np.pi / 3.0
      # ss = np.sqrt(2.0)
      mat = T.set_subtensor(mat[:, :, :], T.stacklists([
        [T.cos(angle), T.sin(angle), -(T.cos(angle) * _w + T.sin(angle) * _h - _w) / (2.0 * _w)],
        [-T.sin(angle), T.cos(angle), -(-T.sin(angle) * _w + T.cos(angle) * _h - _h) / (2.0 * _h)],
        [constv(0, num_batch, self.dtype), constv(0, num_batch, self.dtype), constv(1, num_batch, self.dtype)]]).dimshuffle(2, 0, 1))
      # return [mat, _w, _h]
    elif self.method == 'all':
      mat = T.reshape(para, [-1, 3, 3])
      mat = T.set_subtensor(mat[:, 0, 2], mat[:, 0, 2] / T.cast(width, dtype))
      mat = T.set_subtensor(mat[:, 1, 2], mat[:, 1, 2] / T.cast(height, dtype))
      mat = T.set_subtensor(mat[:, 2, 0], mat[:, 2, 0] * T.cast(width, dtype))
      mat = T.set_subtensor(mat[:, 2, 1], mat[:, 2, 1] * T.cast(height, dtype))
    else:
      raise Exception('method not understood.')
    return transform_affine(mat, input, self.method, scale_factor = self.scale_factor)

def make_rotation_map(sz, agl, w = 0):
    mat = np.zeros((sz, sz), dtype = np.float32)
    if w == 0:
      w = int(sz * 0.3)
    mat[(sz // 2) - w // 2 : (sz // 2) + w // 2, :] = 1.0
    W = generateT(agl, 1.0, [0, 0], [0, 0], [sz, sz, sz, sz])[0]
    return cv2.warpPerspective(mat, W, (sz, sz))

class CastingLayer(lasagne.layers.Layer):
  def __init__(self, incoming, type, **kwargs):
    super(CastingLayer, self).__init__(incoming, **kwargs)
    self.type = type

  def get_output_for(self, inputs, **kwargs):
    return T.cast(inputs, self.type);

  def get_output_shape_for(self, input_shapes):
    return input_shapes

class MahalLayer(lasagne.layers.base.MergeLayer):
  def __init__(self, U, V, M, nb, **kwargs):
    super(MahalLayer, self).__init__([U, V], **kwargs)
    self.M = T.constant(M)
    self.nb = nb
  def get_output_for(self, inputs, **kwargs):
    return T.sum(T.dot(inputs[0], self.M) * T.extra_ops.to_one_hot(inputs[1], self.nb), axis = 1)
  def get_output_shape_for(self, input_shape):
    shp = input_shape[0]
    return (shp[0], 1)


