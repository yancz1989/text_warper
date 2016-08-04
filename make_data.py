# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-27 22:54:00
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-08-03 08:15:50
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import PIL
from PIL import ImageDraw, Image, ImageFont
import scipy as sp
import os
import os.path
import sys
import h5py as h5
import cv2
from draw import *

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def read_poem(path):
  poems = []
  with open(path) as f:
    lines = [unicode(line.decode('utf8')) for line in f]
    for i in range(0, len(lines), 2):
      poem = {}
      poem['title'] = lines[i][0 : lines[i].index(u'《')]
      poem['author'] = lines[i][lines[i].index(u'《') + 1 : lines[i].index(u'》')]
      poem['content'] = [lines[i + 1][j * 8 : (j + 1) * 8] for j in range(4)]
      poems.append(poem)
  return poems

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

def generateImg(M, img, imgS):
  width = np.int32(imgS[1] * 0.95)
  rect = [[(0, 0), (width, 5)], [(0, 8), (width, 13)], [(0, 18), (width, 23)]]
  for i in range(1):
    cv2.rectangle(img, rect[i][0], rect[i][1], color = (255), thickness = -3)
  img = cv2.warpPerspective(img, M, (imgS[2], imgS[3]))
  return img

def make_persps(cnt, flags, scale, cls, imgS):
  if flags[0] == 1:
    scales = np.random.rand(cnt) * (1 - scale) * 2 + scale
  else:
    scales = np.ones(cnt)
  if flags[1] == 1:
    thetas = (np.random.rand(cnt) - 0.5) * np.pi * cls / 180
  else:
    thetas = np.zeros(cnt)
  labels = np.array(thetas / np.pi * 180 + cls / 2).astype(np.int32)
  if flags[2] == 1:
    Ts = 2 * (np.random.rand(cnt, 2) - 0.5)
  else:
    Ts = np.zeros((cnt, 2))

  if flags[3] == 1:
    # Ps = np.hstack((np.random.rand(cnt, 1).reshape(cnt, 1), np.zeros((cnt, 1))))
    # Ps = Ps * 1e-3
    Ps = np.random.rand(cnt, 2) / (3 * imgS[2])
  else:
    Ps = np.zeros((cnt, 2))


  Ms = [generateT(theta, scale, T, P, imgS)
    for (theta, scale, T, P) in zip(thetas, scales, Ts, Ps)]
  Ts = np.asarray([[M[1], M[2]] for M in Ms])
  Ms = np.asarray([M[0] for M in Ms])
  return (Ms, scales, thetas, labels, Ts, Ps)

def make_data(root, cnt, imgS, cls = 0, flags = [1, 1, 1, 1]):
  '''
    cnt: sample count for each poem/text
    imgS: list of paras, [in_w, in_h, out_w, out_h]
  '''
  poems = read_poem('data/poem7.txt')
  # pcnt = len(poems)
  # cnt = cnt * poems
  w = imgS[2]
  h = imgS[3]
  iw = imgS[0]
  ih = imgS[1]
  fsize = 12
  hline = 30
  scale = 0.9
  mkdir(root)
  mkdir(root + 'gt/')
  mkdir(root + 'imgs/')
  fnt = ImageFont.truetype('/home/yancz/text_generator/font/Songti.ttc', fsize)

  AMs = []
  ATs = []
  APs = []
  Ascales = []
  Athetas = []
  Alabels = []
  Abox = []

  for i, poem in zip(range(len(poems)), poems):
    if i % 10 == 9:
      print('finished %d ' % (i + 1))
    mkdir(root + 'imgs/' + str(i) + '/')
    mkdir(root + 'gt/' + str(i) + '/')

    Ms, scales, thetas, labels, Ts, Ps = make_persps(cnt, flags, scale, cls, imgS)

    img = Image.new("RGB", (iw, ih), "black")
    gt = Image.new("RGB", (iw, ih), "black")
    text = [poems[i]['title'], poems[i]['author']] + poems[i]['content']
    box = draw_text(img, gt, [0, iw, 0, ih], 1, fnt, fsize, hline, text)
    img.save(root + 'imgs/' + str(i) + '.jpg')
    gt.save(root + 'gt/' + str(i) + '.jpg')
    # img_warp = img.transform((w, h), Image.PERSPECTIVE, M.flatten(), Image.BICUBIC)
    boxes = []
    ss = np.sqrt(2.0)
    for j, M, P, theta in zip(range(cnt), Ms, Ps, thetas):
      tM = np.array(generateT(-theta, 1.0, [0.0, 0.0], [0.0, 0.0],
        [imgS[2], imgS[3], imgS[2] * np.sqrt(2.0), imgS[3] * np.sqrt(2.0)])[0])
      tP = np.eye(3)
      tP[2, 0 : 2] = -P
      _box = tM.dot(tP).dot(M).dot(box).T.astype('float32')
      boxes.append(np.array([_box[0, 0] / _box[0, 2], _box[0, 1] / _box[0, 2],
        _box[1, 0] / _box[1, 2], _box[1, 1] / _box[1, 2]], dtype='float32'))
      wpgt = cv2.warpPerspective(np.array(gt), M, (w, h))
      cv2.imwrite(root + 'imgs/' + str(i) + '/' + str(j) + '.jpg',
        cv2.warpPerspective(np.array(img), M, (w, h)))
      cv2.imwrite(root + 'gt/' + str(i) + '/' + str(j) + '.jpg',
        wpgt)
      cv2.imwrite(root + 'gt/' + str(i) + '/' + str(j) + '_.jpg',
        cv2.warpPerspective(np.array(wpgt), tM.dot(tP), (int(w * ss), int(h * ss))))

    Abox.append(boxes)
    AMs.append(Ms)
    ATs.append(Ts)
    APs.append(Ps)
    Ascales.append(scales)
    Athetas.append(thetas)
    Alabels.append(labels)
    
  if os.path.exists(root + 'meta.h5'):
    os.remove(root + 'meta.h5')
  print(root + 'meta.h5')
  meta = h5.File(root + 'meta.h5', 'w')
  meta.create_dataset('scales', data = np.array(Ascales, dtype = 'float32'))
  meta.create_dataset('Ts', data = np.array(ATs, dtype = 'float32'))
  meta.create_dataset('Ps', data = np.array(APs, dtype = 'float32'))
  meta.create_dataset('thetas', data = np.array(Athetas, dtype = 'float32'))
  meta.create_dataset('Ms', data = np.array(AMs, dtype = 'float32'))
  meta.create_dataset('labels', data = np.array(Alabels, dtype = 'int32'))
  meta.create_dataset('boxes', data = np.array(Abox, dtype = 'float32'))
  meta.close()


# tagl = triangle(Image.new('RGBA', (l, l)))
# clc = circle(Image.new('RGBA', (l, l)))
# sqr = square(Image.new('RGBA', (l, l)))
# sqrX = square_cross(Image.new('RGBA', (l, l)))
# sqr.save('./sqrt.png')
# tagl.save('./tagl.png')
# clc.save('./clc.png')
# sqrX.save('./sqrtX.png')


'''
parameters:
1. count
2. scale
3. rotation
4. translation
5. perspective
6. path
7. angle range
'''

if __name__ == '__main__':
  np.random.seed(2012310818)
  cnt = int(sys.argv[1])
  imgS = [180, 180, 224, 224]
  # scale, rotation, translation, perspective
  lopt = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]

  if lopt[1] != 0:
    make_data('/home/yancz/text_generator/data/' + sys.argv[6] + '/', cnt, imgS,
      cls = int(sys.argv[7]), flags = lopt)
  else:
    make_data('/home/yancz/text_generator/data/' + sys.argv[6] + '/', cnt, imgS, flags = lopt)
