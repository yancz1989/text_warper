# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-10-20 22:36:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-17 17:22:25
from __future__ import division, print_function, absolute_import, unicode_literals

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

import cv2
import h5py as h5
import sys
import PIL

from PIL import ImageDraw, Image, ImageFont

import os
import os.path
from tools import encode_image, decode_image
import json
from matplotlib import rc

import shutil
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.15e')

root = './'
def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

total = 100.0
msize = 84
sep = msize / 2
h, w = [1697, 2400]
psrc = np.array([[sep, sep], [sep, h - sep], [w - sep, h - sep], [w - sep, sep]])

def meta_go(root):
  flist = os.listdir(root)
  for f in flist:
    with open(root + f, 'r') as jf:
      dat = json.load(jf)
      
    marks = np.array(dat['marks'])
    warp = cv2.findHomography(marks, psrc)
    dat['warp'] = warp[0].tolist()
    with open(root + f, 'w') as jf:
      json.dump(dat, jf)
      
def img2h5(root, path, suffix, shape):
  fpatchs = [f for f in os.listdir(root + path + '/') if f[0] != '.']
  with h5.File(root + 'dat_' + path + '_' + suffix + '.h5', 'w') as dat:
    for fpatch in fpatchs:
      dat[fpatch[:-4]] = encode_image(root + path + '/' + fpatch, shape)

def rtransform(I, pos, rangeA, rangeP):
  def affM(idx, val):
    M = np.eye(3)
    for i, j in enumerate(idx):
      M[j // 3, j % 3] = val[i]
    return M
  
  def rotate(M, theta):
    return affM([0, 1, 3, 4], [np.cos(theta),
      np.sin(theta), -np.sin(theta), np.cos(theta)]).dot(M)
  
  def scale(M, alpha):
    return affM([0, 4], [alpha for i in range(2)]).dot(M)
  
  def translate(M, T):
    return affM([2, 5], [T[0], T[1]]).dot(M)
  
  def rotateC(M, theta, shp):
    return translate(rotate(translate(M,
        [-shp[0] / 2, -shp[1] / 2]), theta), [shp[0] / 2, shp[1] / 2])
  
  def scaleC(M, alpha, shp):
    return translate(scale(translate(M, [-shp[0] / 2, -shp[1] / 2]), alpha),
          [shp[0] / 2, shp[1] / 2])
  
  def perspective(M, P):
    return affM([6, 7], [P[0], P[1]]).dot(M)
  
  x1, y1 = pos
  size_in = np.array([w * 0.18, h * 0.3])
  x2, y2 = pos + size_in
  
  # calculate translate factor, and generate T
  r1 = 256.0 * 0.63 / np.max(size_in)
  size_in *= r1
  T = affM([0, 4, 2, 5], [r1, r1, -x1 * r1 + (256 - size_in[0]) / 2,
                          -r1 * y1 + (256 - size_in[1]) / 2])
  
  theta = np.random.rand() * np.pi * rangeA - np.pi * rangeA / 2
  
  T = rotateC(T, theta, [256, 256])
  Ppara = (np.random.rand(2) - 0.5) * rangeP * 1e-3
  
  tmp = perspective(T, Ppara)
  
  pts = tmp.dot(np.array([[x1, y1, 1], [x1, y2, 1],
                          [x2, y1, 1], [x2, y2, 1]]).astype(np.float32).T)
  pts[0, :] /= pts[2, :]
  pts[1, :] /= pts[2, :]
  xyl = np.array([np.min(pts[0, :]), np.max(pts[0, :]),
                  np.min(pts[1, :]), np.max(pts[1, :])])
  
  alpha = 220.0 / max(xyl[3] - xyl[2], xyl[1] - xyl[0]) * (1 - np.random.rand() * 0.1)
  xyl *= alpha
  
  tmp = perspective(scaleC(T, alpha, [256, 256]), Ppara)
  
  pts = tmp.dot(np.array([[x1, y1, 1], [x1, y2, 1],
                          [x2, y1, 1], [x2, y2, 1]]).astype(np.float32).T)
  pts[0, :] /= pts[2, :]
  pts[1, :] /= pts[2, :]
  xyl = np.array([np.min(pts[0, :]), np.max(pts[0, :]),
                  np.min(pts[1, :]), np.max(pts[1, :])])
  txy = [128 - (xyl[3] + xyl[2]) / 2, 128 - (xyl[1] + xyl[0]) / 2]

  P = perspective(translate(scaleC(T, alpha, [256, 256]), txy), Ppara)
  return [P, theta, alpha, txy, Ppara]

def info(fname):
  tmp = fname.split('_')
  return [0 if tmp[0] == 'cap' else 1, str(tmp[1]), str(tmp[2])]

def generate_sample(fname, pts, jsobj):
  I = cv2.imread(root + 'origin/' + fname + '.jpg').astype(np.float32)
  Igt = cv2.imread(root + 'gt/' + fname + '.jpg').astype(np.float32)
  Imask = cv2.imread(root + 'mask/' + fname + '.jpg').astype(np.float32)
  
  with open(root + 'meta/' + fname + '.json') as f:
    dat = json.load(f)
    Mi = np.array(dat['warp'])
  rangeA = int(sys.argv[1]) / 180.0
  rangeP = float(sys.argv[2])
  for i in range(6):
    for j in range(8):
      [P, theta, alpha, txy, Ppara] = rtransform(I, [pts[i][0], pts[i][1]], rangeA, rangeP)
      
      It = cv2.warpPerspective(I, P.dot(Mi), (256, 256), flags = cv2.INTER_LINEAR)
      Itgt = cv2.warpPerspective(Igt, P, (256, 256), flags = cv2.INTER_LINEAR)
      Itmask = cv2.warpPerspective(Imask, P, (256, 256), flags = cv2.INTER_LINEAR)
      cv2.imwrite(root + 'patch/' + fname + '_' + str(i) + '_' + str(j) + '.jpg', It)
      cv2.imwrite(root + 'pgt/' + fname + '_' + str(i) + '_' + str(j) + '.jpg', Itgt)
      cv2.imwrite(root + 'pmask/' + fname + '_' + str(i) + '_' + str(j) + '.jpg', Itmask)
      jsobj[fname + '_' + str(i) + '_' + str(j)] = [P.tolist(), theta, alpha, txy, Ppara.tolist()]

def sampling():
  if int(sys.argv[1]) == 120:
    np.random.seed(2012310818)
  pts = np.array([(w * 0.1, h * 0.1), (w * 0.1, h * 0.54), (w * 0.4, h * 0.1),
                  (w * 0.4, h * 0.54), (w * 0.7, h * 0.1), (w * 0.7, h * 0.54)], dtype=np.float32)

  jsobj = {}
  flist = os.listdir(root + 'origin/')
  for i in range(len(flist)):
    generate_sample(flist[i][0 : len(flist[i]) - 4], pts, jsobj)

  cap = ['2_17_2', '1_21_5', '1_17_1', '2_18_3', '2_21_5']
  txt = ['0_77_5', '1_77_3', '1_77_4', '1_77_5']

  for f in cap:
    for i in range(8):
      if os.path.exists(root + 'pmask/cap_' + f + '_' + str(i) + '.jpg'):
        os.remove(root + 'pmask/cap_' + f + '_' + str(i) + '.jpg')
        
  for f in txt:
    for i in range(8):
      if os.path.exists(root + 'pmask/txt_' + f + '_' + str(i) + '.jpg'):
        os.remove(root + 'pmask/txt_' + f + '_' + str(i) + '.jpg')
        
  for f in cap:
    for i in range(8):
      if os.path.exists(root + 'pgt/cap_' + f + '_' + str(i) + '.jpg'):
        os.remove(root + 'pgt/cap_' + f + '_' + str(i) + '.jpg')
        

  for f in txt:
    for i in range(8):
      if os.path.exists(root + 'pgt/txt_' + f + '_' + str(i) + '.jpg'):
        os.remove(root + 'pgt/txt_' + f + '_' + str(i) + '.jpg')
        
  for f in cap:
    for i in range(8):
      if os.path.exists(root + 'patch/cap_' + f + '_' + str(i) + '.jpg'):
        os.remove(root + 'patch/cap_' + f + '_' + str(i) + '.jpg')
        

  for f in txt:
    for i in range(8):
      if os.path.exists(root + 'patch/txt_' + f + '_' + str(i) + '.jpg'):
        os.remove(root + 'patch/txt_' + f + '_' + str(i) + '.jpg')

  keys = jsobj.keys()
  js_ = jsobj
  print(keys[0 : 10])
  for f in cap:
    for i in range(8):
      if 'cap_' + f + '_' + str(i) in keys:
        del js_['cap_' + f + '_' + str(i)]
        
        
  for f in txt:
    for i in range(8):
      if 'txt_' + f + '_' + str(i) in keys:
        del js_['txt_' + f + '_' + str(i)]

  with open(root + 'parameter_' + sys.argv[1] + '.json', 'w') as f:
    json.dump(js_, f)
    
  flist = os.listdir(root + 'pmask/')
  for f in flist:
    if f[0] != '.':
      img = cv2.imread(root + 'pmask/' + f) - 255
      if np.count_nonzero(img) == 0:
        print(f)
        
  tlist = [f[:len(f) - 4] for f in os.listdir(root + 'warp/')]

  caps = [t for t in tlist if t[0] == 'c']
  txts = [t for t in tlist if t[0] == 't']
  caps.sort()
  txts.sort()

  trainList = caps[0 : int(len(caps) * 0.8)] + txts[0 : int(len(txts) * 0.8)]
  testList = (caps[int(len(caps) * 0.8) + 1 : int(len(caps) * 0.9)] 
              + txts[int(len(txts) * 0.8) + 1 : int(len(txts) * 0.9)])
  valList = caps[int(len(caps) * 0.9) + 1 : int(len(caps))] + txts[int(len(txts) * 0.9) + 1 : int(len(txts))]

  hsh = {}

  for train in trainList:
    hsh[train] = 'train'
  for test in testList:
    hsh[test] = 'test'
  for val in valList:
    hsh[val] = 'val'

  hshset = {}
  hshset['train'] = []
  hshset['test'] = []
  hshset['val'] = []

  with open(root + 'parameter_' + sys.argv[1] + '.json', 'r') as f:
    meta = json.load(f)
    
  for k, v in meta.items():
    if hsh.get(k[0 : k.rfind('_') - 2]) != None:
      hshset[hsh[k[0 : k.rfind('_') - 2]]].append(k)

  with open(root + 'slices.json', 'w') as f:
    json.dump(hshset, f)

  img2h5(root, 'pmask', str(int(sys.argv[1])), (256, 256))
  img2h5(root, 'patch', str(int(sys.argv[1])), (256, 256))

if __name__ == '__main__':
  sampling()