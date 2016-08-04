# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-03-19 15:08:02
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-11 10:55:18
from __future__ import absolute_import
import sys
import os
import time
import codecs
import re

from PIL import Image, ImageFont, ImageDraw, ImageOps
import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
import scipy as sp

import cv2

import h5py as h5


def list_file(root):
    return [f for f in listdir(root) if (isfile(join(root, f)) and f[0] != '.')]

def mkdir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)
# theta: rotation angle
# scale: scaling factor
# T: translation parameter, which is a random number in [-1, 1] as the ratio to the boundary
# P: perspective
def generateT(theta, scale, T, P, imgS):
    W = np.eye(3)
    W[0 : 2, 2] = np.array([imgS[1], imgS[0]]) / -2.0
    W[2, 0 : 2] = np.array([P[0], P[1]])
    sf = np.sqrt(scale)
    W = np.array([[sf * np.cos(theta), sf * np.sin(theta), 0],
                  [sf * -np.sin(theta), sf * np.cos(theta), 0],
                  [0, 0, 1]]).dot(W)
    W[0 : 2, 2] += np.array([imgS[3], imgS[2]]) / 2.0
    pts = W.dot(np.array([[0, 0, 1], [imgS[1], 0, 1], [0, imgS[0], 1], [imgS[1], imgS[0], 1]]).T)
    tx = np.min([np.min(pts[0, :]), np.min(imgS[3] - pts[0, :])]) * T[0]
    ty = np.min([np.min(pts[1, :]), np.min(imgS[2] - pts[1, :])]) * T[1]
    W[0 : 2, 2] += np.array([tx, ty])
    return W

def resize_img(img, scl):
    return cv2.resize(img, (0,0), fx = scl, fy = scl)

def warp(img, bg, W):
    imgw = cv2.warpPerspective(img, W, (bg.shape[1], bg.shape[0]), borderValue = 0)
    imgp = cv2.warpPerspective(np.ones(img.shape) * 255, W, (bg.shape[1], bg.shape[0]), borderValue = 0)
    bgi = np.empty_like(bg)
    bgi[:] = bg
    bgi[np.where(imgp > 0)] = 0
    imgw += bgi
    return imgw

def array_replicate(arr, cnt):
    for i in range(cnt):
        for s in arr:
            yield s

def make_lines():
    with codecs.open('../data/meta/sentence.txt', 'r', encoding = 'utf-8') as f:
        data = re.split(u'\n|。|？|\?|！|\!', f.read().replace(u'\u3000', '').replace(
            u' ', '').replace(u',', u'，').replace(u'?', u'？').replace(u'.', u'。'))

    all = [s + u'。' for s in data if (
        len(s) > 5 and (not s[0] in [str(i) for i in range(10)]) and (u'-' not in s))]

    imgs = []
    for s in all:
        fsize = int(204.0 / len(s))
        img = Image.new("RGB", (fsize * len(s), fsize + 3), "black")
        f = ImageFont.truetype('../data/fonts/Songti.ttc', fsize)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), s, font = f,
            fill = tuple(np.array([255, 255, 255]).astype(int)))
        imgs.append(np.array(img))

    return imgs

def make_sample(imgs, cnt):
    case = 'line'
    base = '../data/sample/' + case + '/'
    mkdir(base)
    bg = cv2.imread('../data/bg.jpg')
    bg = np.zeros(bg.shape).astype(type(bg[0, 0, 0]))

    l = len(imgs) * cnt
    print('initialize files...')
    f = h5.File(base + case + '.h5', 'w')
    flabel = f.create_dataset('label', (l,), dtype = np.float32)
    fWs = f.create_dataset('Ws', (l, 3, 3), dtype = np.float32)
    fTheta = f.create_dataset('theta', (l, ), dtype = np.float32)
    fScale = f.create_dataset('scale', (l, ), dtype = np.float32)

    imgs = [img for i in range(cnt) for img in imgs]

    # generater
    labels = rnd.randint(0, 60, size = l)
    thetas = (labels * 3.0 + (rnd.rand(l) - 0.5) * 2.0) / 360.0 * np.pi
    scales = np.array([rnd.rand(cnt) * (np.min(bg.shape[0 : 2]) / LA.norm(img.shape) - 0.8) + 0.8
                for img in imgs]).flatten()
    Ts = (rnd.rand(l, 2) - 0.5) * 2
    Ws = [generateT(theta, scale, T, [0,0], [img.shape[0], img.shape[1], bg.shape[0], bg.shape[1]]) for (theta, scale, T, img) in zip(thetas, scales, Ts, imgs)]
    print len(Ws)
    imgW = (warp(img[:, :, :], bg, W) for (img, W) in zip(array_replicate(imgs, cnt), Ws))

    print('generate data... %d %d...' % (l, cnt))
    for i in range(l):
        flabel[i] = labels[i]
        # print fWs[i].shape, Ws[i].shape
        # print type(Ws[i]), type(Ws[i][0, 0])
        # print Ws[i]
        fWs[i, :] = Ws[i].astype(np.float32)
        fTheta = thetas[i]
        fScale = scales[i]
        f.flush()
        imgw = imgW.next()
        cv2.imwrite(base + str(i) + '.jpg', imgw)

if __name__ == '__main__':
    rnd.seed(2012310818)
    # make_sample(sys.argv[1], 200)
    mkdir('../data/lines')
    imgs = make_lines()
    make_sample(imgs, cnt = 200)

