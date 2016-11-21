# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-10-21 10:41:07
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-10-30 11:53:26

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np
import theano
import theano.tensor as T
import lasagne
from train import load_data, iterate_minibatch, read_image
from model import log
import sys
from adam import adam
from collections import OrderedDict

def exp_raw(dtype):
  shp = (None, 3, 256, 256)
  input_var = T.tensor4('input_var', dtype = 'float32')
  psp = T.dmatrix("psp")
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
  network['norm1_1'] = lasagne.layers.BatchNormLayer(network['pool1_1'])

  network['conv1_3'] = lasagne.layers.Conv2DLayer(network['norm1_1'],
    num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['conv1_4'] = lasagne.layers.Conv2DLayer(network['conv1_3'],
    num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool1_2'] = lasagne.layers.MaxPool2DLayer(network['conv1_4'], pool_size = (2, 2))
  network['norm1_2'] = lasagne.layers.BatchNormLayer(network['pool1_2'])

  network['conv1_5'] = lasagne.layers.Conv2DLayer(network['norm1_2'],
    num_filters = 256, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool1_3'] = lasagne.layers.MaxPool2DLayer(network['conv1_5'], pool_size = (2, 2))

  network['conv1_6'] = lasagne.layers.Conv2DLayer(network['pool1_3'],
    num_filters = 256, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
  network['pool1_4'] = lasagne.layers.MaxPool2DLayer(network['conv1_6'], pool_size = (2, 2))

  # Perspective Transform
  network['norm2'] = lasagne.layers.BatchNormLayer(network['pool1_4'])
  # network['cast'] = CastingLayer(network['norm2'], dtype)
  theano.config.floatX = dtype 
  network['pfc2_1'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['norm2'], p = 0.05),
    num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
  network['pfc2_2'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['pfc2_1'], p=0.05),
    num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
  network['pfc2_3'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['pfc2_2'], p=0.05),
    num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
  # loss target 2
  network['pfc_out'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['pfc2_3'], p = 0.05),
    num_units = 8, nonlinearity = lasagne.nonlinearities.rectify)
  theano.config.floatX = 'float32'

  predict = lasagne.layers.get_output(network['pfc_out'])
  loss = T.sqrt(lasagne.objectives.squared_error(predict, psp).mean())
  paras = lasagne.layers.get_all_params(network['pfc_out'], trainable = True)
  updates = adam(loss, paras, [theano.shared(np.float32(0.0001)) for i in range(len(paras))])
  ftrain = theano.function([input_var, psp], [loss, predict], updates = updates)

  def get_inputs(meta, batch, path):
    # batchidx = [keys[i] for i in batch]
    input = np.array([read_image(path + 'patch/' + idx + '.jpg', shape = (256, 256))
      for idx in batch]).astype(np.float32)
    seg = np.array([read_image(path + 'pmask/' + idx + '.jpg', shape = (256, 256))
      for idx in batch]).astype(np.float32)
    dat = [meta[key] for key in batch]
    Ps = np.array([np.array(dat[i][0]).flatten()[0 : 8] for i in range(len(batch))])
    for P in Ps:
      P[6 : 8] = (P[6 : 8] + 1e-3) * 1e4
    return input, Ps

  path = '/home/yancz/text_generator/data/real/'
  dat, meta = load_data(path, 10000, False)
  for epoch in range(10):
    loss = 0
    trs = 0
    for batch in iterate_minibatch(dat['train'], 32, len(dat['train'])):
      inputs = get_inputs(meta, batch, path)
      l, valp = ftrain(*inputs)
      log(l)
      print(valp)
      loss += l
      trs += 1
    loss /= trs
    log('loss ' + str(epoch) + ' ' + str(l))
  return ftrain

def exp_singular(dtype):
  theano.config.floatX = dtype 
  network['pfc2_1'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['norm2'], p = 0.05),
    num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
  network['pfc2_2'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['pfc2_1'], p=0.05),
    num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
  network['pfc2_3'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['pfc2_2'], p=0.05),
    num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
  # loss target 2
  network['pfc_out'] = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network['pfc2_3'], p = 0.05),
    num_units = 2, nonlinearity = lasagne.nonlinearities.rectify)

  predict = lasagne.layers.get_output(network['pfc_out'])
  loss = T.sqrt(lasagne.objectives.squared_error(predict, psp).mean())
  paras = lasagne.layers.get_all_params(network['pfc_out'])
  updates = lasagne.updates.adam(loss['train'], paras, learning_rate = 0.0001)
  ftrain = theano.function(inputs, [loss, predict], updates = updates)
  return ftrain

def exp_pxy():
  def trans(x, y, Px, Py):
    p = 1 + Px * x + Py * y
    return [float(x / (p + np.random.randn() * 0)), float(y / (p + np.random.randn() * 0))]

  dtype = 'float32'
  def sampling(w, h, P):
    X = np.random.randint(0, 256, size = P.shape)
    return np.array([[x[0], x[1]] + trans(x[0], x[1], p[0], p[1]) for x, p in zip(X, P)], dtype=dtype)

  np.random.seed(2012310818)
  n = 1000
  P = np.random.rand(n, 2).astype(dtype) * 1e-3
  X = sampling(256, 256, P)
  P *= 10000

  network = OrderedDict()

  if dtype == 'float64':
    input_var = T.dmatrix('input')
    psp_var = T.dmatrix('psp')
  else:
    input_var = T.fmatrix('input')
    psp_var = T.fmatrix('psp')
    
  theano.config.floatX = dtype
  network['input'] = lasagne.layers.InputLayer(shape = (None, 4), input_var = input_var)
  network['fc1'] = lasagne.layers.DenseLayer(network['input'], num_units=1024,
                                             nonlinearity=lasagne.nonlinearities.rectify)
  network['fc2'] = lasagne.layers.DenseLayer(network['fc1'], num_units=1024,
                                             nonlinearity=lasagne.nonlinearities.rectify)
  network['fc3'] = lasagne.layers.DenseLayer(network['fc2'], num_units=2,
                                             nonlinearity=lasagne.nonlinearities.rectify)

  pred = lasagne.layers.get_output(network['fc3'])
  loss = lasagne.objectives.squared_error(psp_var, pred).mean()
  paras = lasagne.layers.get_all_params(network['fc3'])
  updates = lasagne.updates.adam(loss, paras, 0.0001)
  ftrain = theano.function([input_var, psp_var], [loss, pred], updates = updates)

  theano.config.floatX = 'float32'

  def iterate_minibatch(idx, batchsize, l):
    for start in range(0, l, batchsize):
      yidx = idx[start : start + batchsize]
      yield (X[yidx], P[yidx])
      
  idx = np.random.permutation(np.arange(0, n))
  for i in range(100):
    lval = 0
    cnt = 0
    for batch in iterate_minibatch(idx, 32, n):
      lval += ftrain(batch[0], batch[1])[0]
      cnt += 1
    print(i, lval / cnt)

if __name__ == '__main__':
  if sys.argv[1] == 'raw':
    exp_raw(sys.argv[2])
  else:
    exp_singular(sys.argv[2])
