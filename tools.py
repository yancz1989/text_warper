# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-10-29 20:34:20
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-10-30 17:32:58
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import os.path
import numpy as np
import theano
import theano.tensor as T
import lasagne
import cv2
import h5py
import sys
import numpy as np
import json
from collections import OrderedDict

def get_config(fpath):
  with open(fpath) as f:
    config = json.load(f)
  return config

def encode_image(fp, shape = (-1, -1)):
  # print(fp)
  img = cv2.imread(fp)
  if shape[0] == -1:
    shape = img.shape[0 : 2]
  img = np.rollaxis(cv2.resize(img, shape,
        interpolation = cv2.INTER_CUBIC), 2, 0).astype(np.float32) / 256.0
  return img

def decode_image(arr):
  return np.rollaxis(arr * 256, 0, 3).astype(np.uint8)

def get_time_stamp():
  return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def log(s, fname = ''):
  if fname != '':
    with open(fname, 'a') as f:
      f.write(str(s) + '\n')
  print(s)

def generateT(theta, scale, T, P, imgS):
  W = np.eye(3)
  W[0 : 2, 2] = np.array([imgS[1], imgS[0]]) / -2.0
  sf = np.sqrt(scale)
  W = np.array([[sf * np.cos(theta), sf * np.sin(theta), imgS[3] / 2],
                [sf * -np.sin(theta), sf * np.cos(theta), imgS[2] / 2],
                [0, 0, 1]]).dot(W)
  pts = W.dot(np.array([[0, 0, 1], [imgS[1], 0, 1], [0, imgS[0], 1], [imgS[1], imgS[0], 1]]).T)
  tx = np.min([np.min(pts[0, :]), np.min(imgS[3] - pts[0, :])]) * T[0]
  ty = np.min([np.min(pts[1, :]), np.min(imgS[2] - pts[1, :])]) * T[1]
  W[0 : 2, 2] += np.array([tx, ty])
  matP = np.eye(3)
  matP[2, 0 : 2] = P
  W = matP.dot(W)
  return (W, tx, ty)