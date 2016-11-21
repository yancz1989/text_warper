# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-10-29 20:43:37
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-07 09:17:21

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import as_tuple
import cv2
from collections import OrderedDict
from adam import adam
from tools import log
import h5py as h5

class Model(object):
  def __init__(self, config = None):
    if(config != None):
      self.load(config)

  def load(self, config):
    self.input_shp = (config['input'], config['input'])
    self.opt = config['opt']
    self.path = config['path']
    self.learning_rate = np.float32(config['learning_rate'])
    self.ksize = config['ksize']
    self.stride = config['stride']
    self.acc = config['acc']
    self.layer_rep = config['layers']
    self.dtype = config['dtype']
    self.shared = config['share']
    self.store = config['pstore']
    self.anglefc = config['anglefc']
    self.Arange = config['Arange']
    self.kinit = config['init']
    self.batchsize = config['batchsize']
    self.dropout = config['dropout']
    log('Build model with config ksize %s, stride %s, batchsize %d, learning rate at%s.'
       % (str(self.ksize), str(self.stride), self.batchsize, ''.join([' ' + str(k) for k in self.learning_rate])))

    self.seg = T.tensor4("seg", dtype = 'float32')  
    self.input_var = T.tensor4('input_var', dtype = 'float32')
    self.psp_var = T.fmatrix("psp_var")
    self.angle_var = T.ivector("angle_var")
    self.angler_var = T.fvector("angler_var")
    self.inputs = [self.seg, self.psp_var, self.angle_var]

    self.network = OrderedDict()
    self.network['input'] = lasagne.layers.InputLayer(shape = (None, 3) + self.input_shp, input_var = self.input_var)
    self.network['angle'] = lasagne.layers.InputLayer(shape = (None, ), input_var = self.angle_var)
    self.network['angler'] = lasagne.layers.InputLayer(shape = (None, ), input_var = self.angler_var)
    self.network['psp'] = lasagne.layers.InputLayer(shape = (None, 2), input_var = self.psp_var)

    self.fpred = {}
    self.vals = {}
    self.predict = {}
    self.loss = {}
    self.ffeat = {}

    self.optkey = ['', 'pspout', 'angleout', 'mal']
    self.feat = ['conv1_2', 'conv1_4', 'conv1_5', 'conv1_6', 'conv3_1', 'conv3_4']

  def model():
    raise NotImplementedError('method not implemented.')

  def build(self, deterministic = False):
    self.model()
    self.predict['pspout'] = lasagne.layers.get_output(self.network['pspout'], deterministic = deterministic)
    self.predict['pspimg'] = lasagne.layers.get_output(self.network['pspimg'], deterministic = deterministic)
    self.predict['angleout'] = lasagne.layers.get_output(self.network['angleout'], deterministic = deterministic)
    self.predict['angleimg'] = lasagne.layers.get_output(self.network['angleimg'], deterministic = deterministic)
    self.predict['mal'] = lasagne.layers.get_output(self.network['mal'], deterministic = deterministic)

    self.loss['pspout'] = T.sqrt(lasagne.objectives.squared_error(self.predict['pspout'], self.psp_var).mean())
    self.loss['angleout'] = lasagne.objectives.categorical_crossentropy(self.predict['angleout'], self.angle_var).mean()
    # self.loss['mal'] = self.predict['mal'].mean()
    self.loss['mal'] = 0.01 * T.sqrt(lasagne.objectives.squared_error(T.argmax(self.predict['angleout']), self.angle_var).mean())

    self.loss['train'] = 0
    _inputs = [self.input_var]
    for i, _opt in enumerate(self.opt):
      if _opt == 1:
        if i < 3:
          _inputs.append(self.inputs[i])
        self.loss['train'] += self.loss[self.optkey[i]]
        paras = lasagne.layers.get_all_params(self.network[self.optkey[i]], trainable = True)
        print('Loss %s add...' % self.optkey[i])

    if deterministic == False:
      print('Building training functions...')
      lrvars = []
      self.layer_rep.append(len(paras))
      for lr, i in zip(self.learning_rate, range(len(self.layer_rep) - 1)):
        lrvars += [theano.shared(np.float32(lr)) for j in range(self.layer_rep[i], self.layer_rep[i + 1])]
      self.updates, self.update_params = adam(self.loss['train'], paras, lrvars)
      self.ftrain = theano.function(_inputs, [], updates = self.updates)
      print('Building validation functions...')
      self.vals['pspout'] = abs((self.predict['pspout'] - self.psp_var) / self.psp_var).mean()
      self.vals['angleout'] = T.eq(T.argmax(self.predict['angleout'], axis = 1), self.angle_var).mean()
      self.vals['mal'] = self.loss['mal']

      self.fval = theano.function(_inputs, [self.loss['train']] + 
        [self.vals[self.optkey[i]] for i in range(len(self.opt)) if self.opt[i] == 1])
    else:
      print('Building middle output functions...')
      for f in self.feat:
        self.ffeat[f] = theano.function([self.input_var], lasagne.layers.get_output(self.network[f]))

      print('Building prediction functions...')
      for key in self.predict:
        if key != 'mal':
          self.fpred[key] = theano.function([self.input_var], self.predict[key])

  def save(self, epoch, layer):
    paras = lasagne.layers.get_all_param_values(self.network[layer])
    with h5.File(self.path + self.store + str(epoch) + '.h5', 'w') as dat:
      for i, para in zip(range(len(paras)), paras):
        dat[str(i)] = para
      dat.close()

  def save_update(self):
    with h5.File(self.path + 'para/updates.h5', 'w') as dupdate:
      for i, update in enumerate(self.update_params):
        dupdate[str(i)] = update.get_value()
      dupdate.close()

  def recover(self, epoch, layer):
    values = []
    with h5.File(self.path + self.store + str(epoch) + '.h5', 'r') as dat:
      for i in range(len(dat)):
        values.append(dat[str(i)][:])
      params = lasagne.layers.get_all_params(self.network[layer])
      for i, (p, v) in enumerate(zip(params, values)):
        if p.get_value().shape != v.shape:
          print("mismatch: parameter has shape %r but value to set has shape %r, only recover to layer %d." % (
            p.get_value().shape, v.shape, i))
          break
        else:
            p.set_value(v)
      dat.close()

  def recover_update(self):
    if os.path.isfile(self.path + 'para/updates.h5'):
      with h5.File(self.path + 'para/updates.h5', 'r') as dupdate:
        for i in range(len(dupdate.keys())):
          if update_params[i].get_value().shape != dupdate[str(i)][:].shape:
            raise ValueError("mismatch: parameter has shape %r but value to "
                "set has shape %r" % (update_params[i].get_value().shape, dupdate[str(i)][:].shape))
          else:
            self.update_params[i].set_value(dupdate[str(i)][:])
        dupdate.close()
