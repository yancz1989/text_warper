# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-10-29 20:42:00
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-05 17:30:56

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import as_tuple
from model import Model
from tools import generateT
from layers import PerspectiveLayer, CastingLayer, MahalLayer, make_rotation_map

class ModelNaive(Model):
  def __init__(self, config = None):
    Model.__init__(self, config)

  def model(self):
    self.network['conv1_1'] = lasagne.layers.Conv2DLayer(self.network['input'],
      num_filters = 64, filter_size = (3, 3),nonlinearity = lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
    self.network['conv1_2'] = lasagne.layers.Conv2DLayer(self.network['conv1_1'],
      num_filters = 64, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
    self.network['pool1_1'] = lasagne.layers.MaxPool2DLayer(self.network['conv1_2'], pool_size = (2, 2))
    self.network['norm1_1'] = lasagne.layers.BatchNormLayer(self.network['pool1_1'])

    self.network['conv1_3'] = lasagne.layers.Conv2DLayer(self.network['norm1_1'],
      num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
    self.network['conv1_4'] = lasagne.layers.Conv2DLayer(self.network['conv1_3'],
      num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
    self.network['pool1_2'] = lasagne.layers.MaxPool2DLayer(self.network['conv1_4'], pool_size = (2, 2))
    self.network['norm1_2'] = lasagne.layers.BatchNormLayer(self.network['pool1_2'])

    self.network['conv1_5'] = lasagne.layers.Conv2DLayer(self.network['norm1_2'],
      num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
    self.network['pool1_3'] = lasagne.layers.MaxPool2DLayer(self.network['conv1_5'], pool_size = (2, 2))

    self.network['conv1_6'] = lasagne.layers.Conv2DLayer(self.network['pool1_3'],
      num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
    self.network['pool1_4'] = lasagne.layers.MaxPool2DLayer(self.network['conv1_6'], pool_size = (2, 2))
    self.network['norm2'] = lasagne.layers.BatchNormLayer(self.network['pool1_4'])

    theano.config.floatX = self.dtype 
    self.network['pfc2_1'] = lasagne.layers.DenseLayer(
      lasagne.layers.dropout(self.network['norm2'], p = self.dropout),
      num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
    self.network['pfc2_2'] = lasagne.layers.DenseLayer(
      lasagne.layers.dropout(self.network['pfc2_1'], p = self.dropout),
      num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
    self.network['pfc2_3'] = lasagne.layers.DenseLayer(
      lasagne.layers.dropout(self.network['pfc2_2'], p = self.dropout),
      num_units = 1024, nonlinearity = lasagne.nonlinearities.rectify)
    # loss target 2
    self.network['pspout'] = lasagne.layers.DenseLayer(
      lasagne.layers.dropout(self.network['pfc2_3'], p = self.dropout),
      num_units = 2, nonlinearity = lasagne.nonlinearities.rectify)
    theano.config.floatX = 'float32'

    if self.shared:
      self.network['pspT'] = PerspectiveLayer(self.network['norm1_2'], self.network['pspout'], method = 'perspective')
      self.network['conv3_1'] = lasagne.layers.Conv2DLayer(self.network['pspT'], num_filters = 128,
        filter_size=(3,3),nonlinearity = lasagne.nonlinearities.rectify)
      self.network['pool3_1'] = lasagne.layers.MaxPool2DLayer(self.network['conv3_1'], pool_size = (2, 2))
      upper = self.network['pool3_1']
    else:
      self.network['pspT'] = PerspectiveLayer(self.network['input'], self.network['pspout'], method = 'perspective')
      self.network['conv3_1'] = lasagne.layers.Conv2DLayer(self.network['pspT'], num_filters = 64,
        filter_size=(3,3), nonlinearity=lasagne.nonlinearities.rectify)
      self.network['conv3_2'] = lasagne.layers.Conv2DLayer(self.network['conv3_1'], num_filters = 64,
        filter_size=(3,3), nonlinearity=lasagne.nonlinearities.rectify)
      self.network['pool3_1'] = lasagne.layers.MaxPool2DLayer(self.network['conv3_2'], pool_size = (2, 2))
      self.network['conv3_3'] = lasagne.layers.Conv2DLayer(self.network['pool3_1'],
        num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
      self.network['conv3_4'] = lasagne.layers.Conv2DLayer(self.network['conv3_3'],
        num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
      self.network['pool3_2'] = lasagne.layers.MaxPool2DLayer(self.network['conv3_4'], pool_size = (2, 2))
      self.network['conv3_5'] = lasagne.layers.Conv2DLayer(self.network['pool3_2'],
        num_filters = 128, filter_size = (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
      self.network['pool3_3'] = lasagne.layers.MaxPool2DLayer(self.network['conv3_5'], pool_size = (2, 2))
      upper = self.network['pool3_3']

    shape = lasagne.layers.get_output_shape(upper)
    angles = np.arange(0, self.Arange, self.acc)

    kh = self.ksize
    kw = self.ksize
    stride = (self.stride, self.stride)
    
    if self.kinit == 1:
      W = np.zeros((len(angles), 128, kh, kw), dtype=np.float32)
      mid = (angles[0] + angles[-1]) / 2
      for i, angle in enumerate(angles):
        W[i, :, :] = make_rotation_map(kh, (np.float(angle) - mid) * np.pi / 180)
      W = W * ((np.random.rand(len(angles), shape[1], kh, kw) - 0.5)
          * np.sqrt(2.0 / (shape[1] * kw * kh))).astype(np.float32)
      self.network['conv3_4'] = lasagne.layers.Conv2DLayer(upper,
        num_filters = len(angles), filter_size = (kw, kh), W = W, stride = stride,
        nonlinearity = lasagne.nonlinearities.rectify)
    else:
      self.network['conv3_4'] = lasagne.layers.Conv2DLayer(upper,
        num_filters = len(angles), filter_size = (kw, kh), stride = stride,
        nonlinearity = lasagne.nonlinearities.rectify)

    if self.anglefc:
      self.network['norm3'] = lasagne.layers.BatchNormLayer(self.network['conv3_4'])
    else:
      self.network['avg3'] = lasagne.layers.GlobalPoolLayer(self.network['conv3_4'])
      self.network['norm3'] = lasagne.layers.BatchNormLayer(self.network['avg3'])

    self.network['fc3'] = lasagne.layers.DenseLayer(self.network['norm3'],
      num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    self.network['angleout'] = lasagne.layers.DenseLayer(self.network['fc3'],
      num_units = len(angles), nonlinearity=lasagne.nonlinearities.softmax)

    lag = len(angles)
    M = np.zeros((lag, lag))
    for i in range(lag):
      for j in range(lag):
        M[i][j] = (i - j) ** 2
    self.network['mal'] = MahalLayer(self.network['angleout'], self.network['angle'], M, len(angles))

    self.network['pspimg'] = PerspectiveLayer(self.network['input'], self.network['pspout'], dtype='float32', method = 'perspective')
    self.network['angleimg'] = PerspectiveLayer(self.network['pspimg'], self.network['angleout'], dtype='float32', method = 'angle')

