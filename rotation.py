# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-04-11 16:34:22
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-06-05 09:58:48

from vgg16 import make_vgg16
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.nonlinearities import softmax
import numpy as np
import cv2

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

def make_rotation_map(sz, agl):
    mat = np.zeros((sz, sz), dtype = np.float32)
    mat[(sz / 2) : (sz / 2) + 2, :] = 1.0
    W = generateT(agl, 1.0, [0, 0], [0, 0], [sz, sz, sz, sz])
    return cv2.warpPerspective(mat, W, (sz, sz))

def make_rotation_net(net, incoming, angles):
    shape = lasagne.layers.get_output_shape(incoming)[1 : 4]
    W = np.zeros((len(angles), shape[0], shape[1] - 2, shape[2] - 2), dtype=np.float32)
    for i in range(len(W)):
        W[i, :, :] = make_rotation_map(shape[1] - 2, angles[i])
    W = W * ((np.random.rand(len(angles), shape[0], shape[1] - 2, shape[2] - 2) - 0.5)
        * np.sqrt(2.0 / (shape[0] + shape[1]) * np.prod(shape[2 : ])) * 2.0).astype(np.float32)
    net['conv6'] = ConvLayer(incoming, len(angles), shape[1] - 2, W = W, pad = 1, flip_filters = False)
    net['gavg'] = GlobalPoolLayer(net['conv6'])
    net['fcl'] = DenseLayer(net['gavg'], num_units=4096)
    # net['fcl'] = DenseLayer(net['conv6'], num_units=4096)
    net['softmax'] = DenseLayer(net['fcl'], num_units=60,nonlinearity = softmax)
    return net

if __name__ == '__main__':
    net = make_vgg16(InputLayer((None, 3, 224, 224)))
    net = make_rotation_net(
        net, net['pool5'], np.arange(0, 180.0, 3) / 360.0 * np.pi)
    for subnet in net:
        print subnet, lasagne.layers.get_output_shape(net[subnet])

    paras = lasagne.layers.get_all_param_values(net['softmax'])
    print sum([para.size for para in paras])
    for para in paras:
        print para.shape
