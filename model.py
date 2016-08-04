# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-26 20:35:17
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-07-30 10:17:46

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np
import theano.tensor as T
import theano
import lasagne
from lasagne.utils import as_tuple
from make_data import generateT
import cv2
from vgg16 import make_vgg16
from collections import OrderedDict

def const(x):
  return T.constant(np.float64(x))

def vector(var, l):
  return T.tile(var, l)

def constv(var, l):
  return T.tile(T.constant(var, dtype = 'float64'), l)

def transform_affine(para, input, method, scale_factor = 1):
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
  input_transformed, idx = _interpolate(input_dim, x_s_flat, y_s_flat, out_height, out_width)

  output = T.reshape(input_transformed, (num_batch, out_height, out_width, num_channels))
  output = T.cast(output.dimshuffle(0, 3, 1, 2), 'float32')  # dimshuffle to conv format
  return output

def _interpolate(im, x, y, out_height, out_width):
  # *_f are floats
  num_batch, height, width, channels = im.shape
  height_f = T.cast(height, 'float64')
  width_f = T.cast(width, 'float64')

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
  return [T.set_subtensor(out[idx, :], output), idx]

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
      scale_factor=1, resize_factor = 1, **kwargs):
    super(PerspectiveLayer, self).__init__(
        [incoming, localization_network], **kwargs)
    self.scale_factor = as_tuple(scale_factor, 2)
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
    mat = T.zeros((num_batch, 3, 3), dtype='float64')
    mat = T.set_subtensor(mat[:, 0, 0], const(1.0))
    mat = T.set_subtensor(mat[:, 1, 1], const(1.0))
    mat = T.set_subtensor(mat[:, 2, 2], const(1.0))

    if self.method == 'perspective':
      mat = T.set_subtensor(mat[:, 2, 0], para[:, 0] * T.cast(width, 'float64') / 1e4)
      mat = T.set_subtensor(mat[:, 2, 1], para[:, 1] * T.cast(height, 'float64') / 1e4)
    elif self.method == 'angle':
      angle = (360 - T.cast(T.argmax(para, axis = 1), 'float64')) * np.pi / 180
      ss = np.sqrt(2.0)
      mat = T.set_subtensor(mat[:, :, :], T.stacklists([
        [ss * T.cos(angle), ss * T.sin(angle), constv(0.0, num_batch)],
        [-ss * T.sin(angle), ss * T.cos(angle), constv(0.0, num_batch)],
        [constv(0, num_batch), constv(0, num_batch), constv(1, num_batch)]]).dimshuffle(2, 0, 1))
    elif self.method == 'all':
      mat = T.reshape(para, [-1, 3, 3])
      mat = T.set_subtensor(mat[:, 0, 2], mat[:, 0, 2] / T.cast(width, 'float64'))
      mat = T.set_subtensor(mat[:, 1, 2], mat[:, 1, 2] / T.cast(height, 'float64'))
      mat = T.set_subtensor(mat[:, 2, 0], mat[:, 2, 0] * T.cast(width, 'float64'))
      mat = T.set_subtensor(mat[:, 2, 1], mat[:, 2, 1] * T.cast(height, 'float64'))
    else:
      raise Exception('method not understood.')
    return transform_affine(mat, input, self.method, self.scale_factor)

def make_rotation_map(sz, agl, w = 0):
    mat = np.zeros((sz, sz), dtype = np.float32)
    if w == 0:
      w = int(sz * 0.3)
    mat[(sz // 2) - w // 2 : (sz // 2) + w // 2, :] = 1.0
    W = generateT(agl, 1.0, [0, 0], [0, 0], [sz, sz, sz, sz])[0]
    return cv2.warpPerspective(mat, W, (sz, sz))

class CastingLayer(lasagne.layers.Layer):
  def __init__(self, incoming, type, **kwargs):
    super(CastingLayer, self).__init__(
        incoming, **kwargs)
    self.type = type

  def get_output_for(self, inputs, **kwargs):
    return T.cast(inputs, self.type);

  def get_output_shape_for(self, input_shapes):
    return input_shapes

def model(shp, angles, ksize, stride, input_var):
  network = OrderedDict()
  network['input'] = lasagne.layers.InputLayer(shape = shp, input_var = input_var)
  # network = make_vgg16(network, 'model/vgg16_weights_from_caffe.h5')
  # First conv and segmentation part
  network['conv1_1'] = lasagne.layers.Conv2DLayer(network['input'],
    num_filters = 64, filter_size = (3, 3),nonlinearity = lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform())
  network['conv1_2'] = lasagne.layers.Conv2DLayer(network['conv1_1'],
    num_filters = 64, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool1_1'] = lasagne.layers.MaxPool2DLayer(network['conv1_2'], pool_size = (2, 2))
  network['conv1_3'] = lasagne.layers.Conv2DLayer(network['pool1_1'],
    num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool1_2'] = lasagne.layers.MaxPool2DLayer(network['conv1_3'], pool_size = (2, 2))
  network['conv1_4'] = lasagne.layers.Conv2DLayer(network['pool1_2'],
    num_filters = 256, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool1_3'] = lasagne.layers.MaxPool2DLayer(network['conv1_4'], pool_size = (2, 2))
  network['conv1_5'] = lasagne.layers.Conv2DLayer(network['pool1_3'],
    num_filters = 256, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool1_4'] = lasagne.layers.MaxPool2DLayer(network['conv1_5'], pool_size = (2, 2))
  network['conv1_6'] = lasagne.layers.Conv2DLayer(network['pool1_4'],
    num_filters = 512, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool1_5'] = lasagne.layers.MaxPool2DLayer(network['conv1_6'], pool_size = (2, 2))

  # Perspective Transform
  network['norm2'] = lasagne.layers.BatchNormLayer(network['pool1_5'])
  network['cast'] = CastingLayer(network['norm2'], 'float64')
  network['pfc2_1'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['cast'], p = 0.05),
    # network['cast'],
    num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
  network['pfc2_2'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['pfc2_1'], p=0.05),
    # network['pfc2_1'],
    num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
  # loss target 2
  network['pfc_out'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['pfc2_2'], p = 0.05),
    # network['pfc2_2'],
    num_units = 2, nonlinearity = lasagne.nonlinearities.rectify)

  # output feature map
  network['pspT'] = PerspectiveLayer(network['input'],
    network['pfc_out'], method = 'perspective')

  # Angle detection and Transform
  network['conv3_1'] = lasagne.layers.Conv2DLayer(network['input'],
    num_filters = 64, filter_size = (3, 3),nonlinearity = lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform())
  network['pool3_1'] = lasagne.layers.MaxPool2DLayer(network['conv3_1'], pool_size = (2, 2))
  network['conv3_2'] = lasagne.layers.Conv2DLayer(network['pool3_1'],
    num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['conv3_3'] = lasagne.layers.Conv2DLayer(network['conv3_2'],
    num_filters = 64, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool3_2'] = lasagne.layers.MaxPool2DLayer(network['conv3_3'], pool_size = (2, 2))
  upper = network['conv3_3']
  shape = lasagne.layers.get_output_shape(upper)
  if ksize == None:
    kh = shape[2] // 2
    kw = shape[3] // 2
  else:
    kh = ksize
    kw = ksize

  if stride == None:
    stride = (1, 1)
  else:
    stride = (stride, stride)
  W = np.zeros((len(angles), shape[1], kh, kw), dtype=np.float32)
  for i, angle in enumerate(angles):
    W[i, :, :] = make_rotation_map(kh,
     (np.float(angle) - len(angles) / 2) * np.pi / 180)
  W = W * ((np.random.rand(len(angles), shape[1], kh, kw) - 0.5)
      * np.sqrt(2.0 / (shape[1] * kw * kh))).astype(np.float32)
  network['conv3_4'] = lasagne.layers.Conv2DLayer(upper,
    num_filters = len(angles), filter_size = (kw, kh), stride = stride,
    nonlinearity = lasagne.nonlinearities.rectify)
  network['avg3'] = lasagne.layers.GlobalPoolLayer(network['conv3_4'])
  network['norm3'] = lasagne.layers.BatchNormLayer(network['avg3'])
  network['fc3'] = lasagne.layers.DenseLayer(network['norm3'],
    num_units=1024, nonlinearity=lasagne.nonlinearities.rectify)
  # loss target 3
  network['angle'] = lasagne.layers.DenseLayer(network['fc3'],
    num_units = len(angles), nonlinearity=lasagne.nonlinearities.softmax)

  # transform according to angle. The last loss target
  network['angleT'] = PerspectiveLayer(network['pspT'], network['angle'], method = 'angle')

  # location detection
  network['conv4_1'] = lasagne.layers.Conv2DLayer(network['angleT'], num_filters = 64,
    filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['conv4_2'] = lasagne.layers.Conv2DLayer(network['conv4_1'], num_filters = 1,
    filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['norm4'] = lasagne.layers.BatchNormLayer(network['conv4_2'])
  network['fc4'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['norm4'], p = 0.5),
    num_units = 256, nonlinearity = lasagne.nonlinearities.rectify)
  network['loc'] = lasagne.layers.DenseLayer(network['fc4'], num_units = 4,
    nonlinearity = lasagne.nonlinearities.rectify)
  return network

def set_layer_parameter(layer, vparams):
  params = layer.get_params()
  for param, v in zip(params, vparams):
    if param.get_value().shape == v.shape:
      param.set_value(v)
    else:
      raise ValueError('Error! shape of parameter and its value doesn\'t match.')

def get_layer_parameter(layer):
  return [p.get_value() for p in layer.get_params()]

def load_model(network, fname):
  with h5.File(fname, 'r') as dat:
    for key in dat:
      set_layer_parameter(network[key], dat[key])

# def save_model(network, fname):
#   for

def build(shp, opt, acc, learning_rate = 0.001, ksize = None, stride = None):
  input_var = T.tensor4('input_var', dtype = 'float32')
  seg = T.tensor4("seg", dtype = 'float32')
  psp = T.dmatrix("psp")
  angle = T.ivector("angle")
  loc = T.fmatrix("loc")

  outlayer = ['conv1_3', 'pfc_out', 'angle', 'loc']
  optkey = ['seg', 'psp', 'angle', 'loc']
  lossl = ['segmentation', 'perspective', 'angle', 'location']
  vars = [seg, psp, angle, loc]

  inputs = [input_var]
  loss = {}
  predict = {}
  fpred = {}
  vals = {}

  network = model((None, 3, shp[0], shp[1]), range(0, 180, acc), ksize, stride, input_var)

  predict['seg'] = lasagne.layers.get_output(network['conv1_3'])
  predict['psp'] = lasagne.layers.get_output(network['pfc_out'])
  predict['pspT'] = lasagne.layers.get_output(network['pspT'])
  predict['angle'] = lasagne.layers.get_output(network['angle'])
  predict['angleT'] = lasagne.layers.get_output(network['angleT'])
  # predict['loc'] = lasagne.layers.get_output(network['loc'])

  loss['train'] = 0
  loss['seg'] = T.sqrt(lasagne.objectives.squared_error(predict['seg'], seg).mean())
  loss['psp'] = T.sqrt(lasagne.objectives.squared_error(predict['psp'], psp).mean())
  loss['angle'] = lasagne.objectives.categorical_crossentropy(predict['angle'], angle).mean()
  # loss['loc'] = T.sqrt(lasagne.objectives.squared_error(predict['loc'], loc).mean())

  for i, _opt in enumerate(opt):
    if _opt == 1:
      inputs.append(vars[i])
      loss['train'] += loss[optkey[i]]
      paras = lasagne.layers.get_all_params(network[outlayer[i]], trainable = True)
      print('Loss %s add...' % optkey[i])

  print('Building training functions...')
  lr = theano.shared(np.float32(learning_rate))
  updates = lasagne.updates.adam(loss['train'], paras, learning_rate = lr)
  ftrain = theano.function(inputs, [], updates = updates)

  predict['angle'] = T.argmax(predict['angle'], axis = 1)
  print('Building validation functions...')
  vals['seg'] = (loss['seg'] / seg).mean()
  vals['psp'] = abs((predict['psp'] - psp) / psp).mean()
  vals['angle'] = T.eq(predict['angle'], angle).mean()
  # vals['loc'] = (loss['loc'] / loc).mean()

  fval = theano.function(inputs, [loss['train']] + 
    [vals[optkey[i]] for i in range(len(opt)) if opt[i] == 1])

  print('Building prediction function...')
  for key in predict:
    fpred[key] = theano.function([input_var], predict[key])
  return network, lr, ftrain, fval, fpred

