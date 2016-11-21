# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-28 16:50:02
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-07 09:14:14
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

class Trainer(object):
  def __init__(self):
    pass

  def warmup(self, fconfig, model, deterministic = False):
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
    self.model.build(deterministic = False)

    with open(config['path'] + 'parameter_' + str(config['Arange']) + '.json', 'r') as f:
      self.meta = json.load(f)
    # with open(config['slice'], 'r') as f:
    #   slices = json.load(f)

    # if self.sliced == False:
      # all = slices['train'] + slices['val'] + slices['test']
      # all = [all[i] for i in np.random.permutation(len(all))]
      # self.train_idx = all[0 : int(self.samples * 0.8)]
      # self.val_idx = all[int(self.samples * 0.8) : int(self.samples * 0.9)]
      # self.test_idx = all[int(self.samples * 0.9) : self.samples]
    #   with open(config['path'] + 'unsliced.json') as f:
    #     self.dat_idx = json.load(f)
    #   print('not sliced')
    # else:
      # self.train_idx = [slices['train'][i] for i in
      #      np.random.permutation(len(slices['train']))[0 : int(self.samples * 0.8)]]
      # self.val_idx = [slices['val'][i] for i in
      #       np.random.permutation(len(slices['val']))[0 : int(self.samples * 0.1)]]
      # self.test_idx = [slices['test'][i] for i in
      #       np.random.permutation(len(slices['test']))[0 : int(self.samples * 0.1)]]
      # with open
      # print('sliced')

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
    angles = np.array([np.int32((dat[i][1] / np.pi * 180 + self.Arange / 2) / self.acc)
      for i in range(len(batch))])
    return input, seg, Ps, angles

  def iterate_minibatch(self, idx, batchsize, l):
    ridx = np.random.permutation(range(0, l, batchsize))
    for start in ridx:
      yield idx[start : start + batchsize]

  def train(self):
    lidx = len(self.idx) + 1
    with h5.File(self.tmp_dir + 'loss.h5', 'w') as f:
      f['loss'] = -1.0 * np.ones((10000, lidx * 2))

    start = 1
    if self.paraf != '':
      self.paraf = int(self.paraf)
      self.model.recover(self.paraf, 'angleout')
      start = self.paraf + 1
      log('load parameters from %s, start from epoch %d.' % (self.paraf, start), self.logf)
    if self.recoveru:
      self.model.recover_update()
    innerT = 0
    innerV = 0
    if self.opt[3] == 1:
      self.idx = self.idx[:-1]

    print('start training...')
    for epoch in range(start, self.epochs + start):
      # loss[0] train loss, loss[1 : lidx + 1] train loss for each target, else validate
      loss = np.zeros(lidx * 2)
      trs = 0
      vls = 0
      err = 0
      start = time.time()

      for batch in self.iterate_minibatch(self.train_idx, self.batchsize, len(self.train_idx)):
        inputs = self.get_inputs(batch)
        self.model.ftrain(*[inputs[i] for i in ([0] + self.idx)])
        vvals = np.array(self.model.fval(*[inputs[i] for i in ([0] + self.idx)]))
        log('train ' + str(innerT) + ''.join([' ' + str(v) for v in vvals]), self.logf)
        loss[:lidx] += vvals
        trs += 1
        innerT += 1

      for batch in self.iterate_minibatch(self.val_idx, self.batchsize, len(self.val_idx)):
        inputs = self.get_inputs(batch)
        vvals = np.array(self.model.fval(*[inputs[i] for i in ([0] + self.idx)]))
        log('val ' + str(innerV) + ''.join([' ' + str(v) for v in vvals]), self.logf)
        loss[lidx: 2 * lidx] += vvals
        vls += 1
        innerV += 1 

      loss[:lidx] /= trs
      loss[lidx : 2 * lidx] /= vls
      with h5.File(self.tmp_dir + 'loss.h5') as f:
        f['loss'][epoch, :] = loss
      self.model.save(epoch, 'angleout')

      tpass = time.time() - start
      fmt = ('epoch %d time %f train loss' + ''.join([' %f' for i in range(lidx)])
        + ' valid' + ''.join([' %f' for i in range(lidx)]))
      log(fmt % tuple([epoch, tpass] + list(loss)), self.logf)

  def test():
    pass

if __name__ == '__main__':
  model = ModelNaive()
  trainer = Trainer()
  trainer.warmup(sys.argv[1], model, deterministic = False)
  trainer.train()


