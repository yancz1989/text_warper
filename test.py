# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-28 16:50:02
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-07-22 10:32:21
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
from make_data import read_poem, make_data, mkdir
from model import build

import json

def load_data(path, cnt):
  poems = read_poem('data/poem7.txt')
  meta = h5.File(path + 'meta.h5', 'r')
  idx = np.arange(len(poems) * cnt)
  np.random.shuffle(idx)

  train_idx = idx[0 : int(len(idx) * 0.8)]
  val_idx = idx[int(len(idx) * 0.8): int(len(idx) * 0.9)]
  test_idx = idx[int(len(idx) * 0.9) : len(idx)]

  dat = {'poems' : poems, 'train' : train_idx, 'val' : val_idx, 'test' : test_idx}
  for key in meta:
    dat[key] = meta[key][:]
  meta.close()
  return dat

def iterate_minibatch(idx, batchsize, l):
  for start in range(0, l, batchsize):
    yield np.array(idx[start : start + batchsize], dtype='int32')

def read_image(fp, shape = (-1, -1)):
  img = cv2.imread(fp)
  if shape[0] == -1:
    shape = img.shape[0 : 2]
  img = np.rollaxis(cv2.resize(img, shape, interpolation = cv2.INTER_CUBIC), 2, 0).astype(np.float32)
  for i in range(len(img)):
    img[i, :] = (img[i, :] - np.min(img[i, :])) / (np.max(img[i, :]) - np.min(img[i, :]))
  return img

def get_inputs(batch, path, dat, cnt, shape, acc):
  idxs = np.array(batch / cnt, dtype=np.int32)
  idxs_ = batch % cnt
  input = np.array([read_image(path + 'imgs/' + str(idx) + '/' + str(idx_) + '.jpg', shape = (224, 224)) for idx, idx_ in zip(idxs, idxs_)]).astype(np.float32)
  seg = np.array([read_image(path + 'gt/' + str(idx) + '/' + str(idx_) + '.jpg', shape = (224, 224))
    for idx, idx_ in zip(idxs, idxs_)]).astype(np.float32)
  Ps = np.array(dat['Ps'][idxs, idxs_, :]).astype(np.float64) * 1e4 + 1e-9
  # Ps = np.array(dat['labels'][idxs, idxs_]) // acc
  angles = np.array(dat['labels'][idxs, idxs_]) // acc
  box = np.array(dat['boxes'][idxs, idxs_, :]).astype(np.float32)
  return seg, input, Ps, angles, box

def get_time_stamp():
  return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def save_parameter(network, path, epoch):
  paras = lasagne.layers.get_all_param_values(network)
  with h5.File(path + str(epoch) + '.h5', 'w') as dat:
    for i, para in zip(range(len(paras)), paras):
      dat[str(i)] = para
    dat.close()

def load_parameter(net, fname):
  paras = []
  with h5.File(fname, 'r') as dat:
    for i in range(len(dat)):
      paras.append(dat[str(i)][:])
    lasagne.layers.set_all_param_values(net, paras)
    dat.close()

def get_config(fpath):
  with open(fpath) as f:
    config = json.load(f)
    input_shp = (config['input'], config['input'])
    opt = config['opt']
    path = config['path']
    learning_rate = np.float32(config['learning_rate'])
    decay = config['decay']
    epochs = config['epochs']
    batchsize = 4
    ksize = config['ksize']
    store = config['pstore']
    paraf = config['pfile']
    stride = config['stride']
    acc = config['acc']
    samples = config['samples']
  return (input_shp, opt, path, learning_rate, decay, epochs, batchsize, ksize, store, paraf, stride, acc, samples)

def test(fpath, epochs):
  input_shp, opt, path, learning_rate, decay, epochs, batchsize, ksize, store, paraf, stride, acc, samples = get_config(fpath)
  print('config: ksize %s, stride %s, batchsize %d.' % (str(ksize), str(stride), batchsize))

  network, lr, ftrain, fvals, predict = build(input_shp, opt, acc, learning_rate, ksize, stride)
  seg_shape = lasagne.layers.get_output_shape(network['conv1_2'])[2 : 4]

  dat = load_data(path, samples)
  idx = [i + 1 for i, t in enumerate(opt) if t == 1]
  lidx = len(idx) + 1
  sops = ['segment', 'perspective', 'angle', 'location']
  print(('%d training samples, %d validation samples, %d test samples...with option: '
    + ''.join([sops[i - 1] + ' ' for i in idx]))
     %  tuple([len(dat['train']), len(dat['val']), len(dat['test'])]))

  tmp_dir = path + store
  mkdir(tmp_dir)

  with h5.File(tmp_dir + 'loss.h5', 'w') as f:
    f['loss'] = -1.0 * np.ones((10000, lidx * 2))

  print('start training...')
  for epoch in epochs:
    paraf = tmp_dir + paraf
    load_parameter(network['pfc_out'], paraf)
    print('load parameters from %s, start from epoch %d.' % (paraf, start))
    # loss[0] train loss, loss[1 : lidx + 1] train loss for each target, else validate
    loss = np.zeros(lidx * 2)
    trs = 0
    vls = 0
    err = 0
    start = time.time()
    for batch in iterate_minibatch(dat['train'], batchsize, len(dat['train'])):
      inputs = get_inputs(batch, path, dat, samples, seg_shape, acc)
      loss[:lidx] += np.array(fvals(*[inputs[i] for i in ([0] + idx)]))
      trs += 1
    for batch in iterate_minibatch(dat['val'], batchsize, len(dat['val'])):
      inputs = get_inputs(batch, path, dat, samples, seg_shape, acc)
      loss[lidx: 2 * lidx] += np.array(fvals(*[inputs[i] for i in ([0] + idx)]))
      vls += 1
    loss[:lidx] /= trs
    loss[lidx : 2 * lidx] /= vls
    tpass = time.time() - start
    fmt = ('epoch %d: time %f train loss:' + ''.join([' %f' for i in range(lidx)])
      + ', valid:' + ''.join([' %f' for i in range(lidx)]))
    print(fmt % tuple([epoch, tpass] + list(loss)))

if __name__ == '__main__':
  np.random.seed(2012310818)
  input_shp = (224, 224)
  test(sys.argv[2], range(10, 20))
