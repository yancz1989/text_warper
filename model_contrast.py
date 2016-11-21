# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-10-29 20:42:26
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-06 19:19:41

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import as_tuple
from model import Model
from adam import adam
from train import Trainer
from tools import generateT
import sys
from layers import PerspectiveLayer, CastingLayer, MahalLayer, make_rotation_map

class ModelAngleRegress(Model):
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

    self.network['pspT'] = PerspectiveLayer(self.network['norm1_2'], self.network['pspout'], method = 'perspective')
    self.network['conv3_1'] = lasagne.layers.Conv2DLayer(self.network['pspT'], num_filters = 128,
      filter_size=(3,3),nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    self.network['pool3_1'] = lasagne.layers.MaxPool2DLayer(self.network['conv3_1'], pool_size = (2, 2))
    upper = self.network['pool3_1']

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

    self.network['norm3'] = lasagne.layers.BatchNormLayer(self.network['conv3_4'])
    self.network['fc3'] = lasagne.layers.DenseLayer(self.network['norm3'],
      num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    self.network['fc4'] = lasagne.layers.DenseLayer(self.network['fc3'],
      num_units=512, nonlinearity=lasagne.nonlinearities.linear)
    self.network['angleout'] = lasagne.layers.DenseLayer(self.network['fc4'],
      num_units = 1,  nonlinearity=lasagne.nonlinearities.linear)

  def build(self, deterministic = False):
    self.model()
    self.predict['pspout'] = lasagne.layers.get_output(self.network['pspout'])
    self.predict['angleout'] = lasagne.layers.get_output(self.network['angleout'])

    self.loss['pspout'] = T.sqrt(lasagne.objectives.squared_error(self.predict['pspout'], self.psp_var).mean())
    self.loss['angleout'] = T.sqrt(lasagne.objectives.squared_error(self.predict['angleout'], self.angler_var).mean())
    self.loss['train'] = self.loss['pspout'] + self.loss['angleout']

    _inputs = [self.input_var, self.psp_var, self.angler_var]
    paras = lasagne.layers.get_all_params(self.network['angleout'], trainable = True)

    print('Building training functions...')
    lrvars = []
    self.layer_rep.append(len(paras))
    for lr, i in zip(self.learning_rate, range(len(self.layer_rep) - 1)):
      lrvars += [theano.shared(np.float32(lr)) for j in range(self.layer_rep[i], self.layer_rep[i + 1])]
    self.updates, self.update_params = adam(self.loss['train'], paras, lrvars)
    self.ftrain = theano.function(_inputs, [], updates = self.updates)

    print('Building validation functions...')
    self.vals['pspout'] = abs((self.predict['pspout'] - self.psp_var) / self.psp_var).mean()
    self.vals['angleout'] = abs((self.predict['angleout'] - self.angler_var) / self.angler_var).mean()
    self.fval = theano.function(_inputs, [self.loss['train']] + 
      [self.vals['pspout'], self.vals['angleout']])


class RegressTrainer(Trainer):
  def __init__(self):
    pass

  def get_inputs(self, batch):
    input = np.array([self.dat_patch[idx][:] for idx in batch])
    seg = None # np.array([self.dat_pmask[idx + '.jpg'] for idx in batch])
    dat = [self.meta[key] for key in batch]
    Ps = np.array([(np.array(dat[i][4]) + 1e-3) * 1e4 for i in range(len(batch))]).astype(np.float32)
    angles = np.array([dat[i][1] for i in range(len(batch))], dtype=np.float32)
    return input, seg, Ps, angles

if __name__ == '__main__':
  model = ModelAngleRegress()
  trainer = RegressTrainer()
  trainer.warmup(sys.argv[1], model)
  trainer.train()
