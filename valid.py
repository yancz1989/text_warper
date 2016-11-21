# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-11-07 09:13:56
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-07 10:08:36
from __future__ import division, absolute_import, print_function, unicode_literals

import sys
import os
import datetime
import time

import numpy as np
import h5py as h5

import cv2

import theano
import theano.tensor as T

import lasagne
from model import Model
from model_naive import ModelNaive
from tools import mkdir, get_config, log, decode_image
import json

class Validater(object):
  def __init__(self):
    pass

  def warmup(self, fconfig, model, deterministic = True):
    np.random.seed(2012310818)
    config = get_config(fconfig)

    self.path = config['path']
    self.paraf = config['pfile']
    self.samples = config['samples']
    self.opt = config['opt']
    self.store = config['pstore']
    self.Arange = config['Arange']
    self.epochs = config['epochs']
    self.sliced = config['sliced']
    self.batchsize = config['batchsize']
    self.decay = config['decay']
    self.acc = config['acc']
    self.recoveru = config['update']
    self.logf = fconfig[:-4] + 'log'
    self.dat_patch = h5.File(config['path'] + 'dat_patch_' + str(config['Arange']) + '.h5')
    self.dat_pmask = h5.File(config['path'] + 'dat_patch_' + str(config['Arange']) + '.h5')

    self.model = model
    self.model.load(config)
    self.model.build(deterministic = True)

    with open(config['path'] + 'parameter_' + str(config['Arange']) + '.json', 'r') as f:
      self.meta = json.load(f)

    with open(config['path'] + ('unsliced.json' if self.sliced == False else 'sliced.json')) as f:
      self.dat_idx = json.load(f)

    self.train_idx = self.dat_idx['train']
    self.val_idx = self.dat_idx['val']
    self.test_idx = self.dat_idx['test']


    self.idx = [i + 1 for i, t in enumerate(self.opt) if t == 1]
    sops = ['segment', 'perspective', 'angle', 'mal']
    print(('%d training samples, %d validation samples, %d test samples...with option: '
      + ''.join([sops[i - 1] + ' ' for i in self.idx]))
       %  tuple([len(self.train_idx), len(self.val_idx), len(self.test_idx)]))
    self.tmp_dir = self.path + self.store 
    mkdir(self.tmp_dir)

  def get_inputs(self, batch):
    input = np.array([self.dat_patch[idx][:] for idx in batch])
    seg = None # np.array([self.dat_pmask[idx + '.jpg'] for idx in batch])
    dat = [self.meta[key] for key in batch]
    Ps = np.array([(np.array(dat[i][4]) + 1e-3) * 1e4 for i in range(len(batch))]).astype(np.float32)
    angles = np.array([dat[i][1] for i in range(len(batch))])
    return input, seg, Ps, angles

  def iterate_minibatch(self, idx, batchsize, l):
    ridx = np.random.permutation(range(0, l, batchsize))
    for start in ridx:
      yield idx[start : start + batchsize]

  def predict(self, idxs):
    output = {}
    for batch in self.iterate_minibatch(idxs, self.batchsize, len(idxs)):
      input = self.get_inputs(batch)
      pspout = self.model.fpred['pspout'](input[0])
      if self.opt[2] == 1:
        angleout = self.model.fpred['angleout'](input[0])
      for i, idx in enumerate(batch):
        output[idx] = pspout[i].tolist() + input[2][i].tolist()
        if self.opt[2] == 1:
           output[idx] = output[idx] + angleout[i].tolist() + [input[3][i]]
    return output

  def calibrate(self, idxs):
    output = {}
    for idx in idxs:
      output[idx] = {}
    for batch in self.iterate_minibatch(idxs, self.batchsize, len(idxs)):
      input = self.get_inputs(batch)
      pspout = self.model.fpred['pspout'](input[0])
      angleout = self.model.fpred['angleout'](input[0])
      pspimg = self.model.fpred['pspimg'](input[0])
      angleimg = self.model.fpred['angleimg'](input[0])
      for i, idx in enumerate(batch):
        output[idx] = [decode_image(input[0][i]), decode_image(pspimg[i]), decode_image(angleimg[i]), pspout[i],
          np.argsort(angleout[i])[::-1], input[2][i], input[3][i]]
    return output

if __name__ == '__main__':
  model = ModelNaive()
  valid = Validater()
  valid.warmup(sys.argv[1], model)
  valid.model.recover(int(sys.argv[2]), 'angleout')
  rtrain = valid.predict(valid.train_idx)
  rtest = valid.predict(valid.test_idx)
  rvalid = valid.predict(valid.val_idx)

  with h5.File(valid.path + valid.store + 'bench.h5', 'w') as dat:
    for key in rtrain:
      dat['train/' + key] = np.array(rtrain[key], dtype=np.float32)
    for key in rtest:
      dat['test/' + key] = np.array(rtest[key], dtype=np.float32)
    for key in rvalid:
      dat['valid/' + key] = np.array(rvalid[key], dtype=np.float32)


