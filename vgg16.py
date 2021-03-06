# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-04-05 19:01:27
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-07-05 11:27:18

from __future__ import division, print_function, unicode_literals, absolute_import
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax
import h5py as h5
import numpy as np

def make_vgg16(net, filename = None, with_fc = False, filter = 3):
    input_layer = net['input']
    net['conv1_1'] = ConvLayer(
        input_layer, 64, filter, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, filter, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, filter, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, filter, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, filter, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, filter, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, filter, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, filter, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, filter, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, filter, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, filter, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, filter, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, filter, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    if with_fc:
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    if filename != None:
        print('Load model data from %s...' % filename)
        vgg16 = h5.File(filename, 'r')
        lasagne_model = [t.astype(np.float32) for t in [
            vgg16['conv1_1#w'][:], vgg16['conv1_1#b'][:], vgg16['conv1_2#w'][:], vgg16['conv1_2#b'][:],
            vgg16['conv2_1#w'][:], vgg16['conv2_1#b'][:], vgg16['conv2_2#w'][:], vgg16['conv2_2#b'][:],
            vgg16['conv3_1#w'][:], vgg16['conv3_1#b'][:], vgg16['conv3_2#w'][:], vgg16['conv3_2#b'][:],
            vgg16['conv3_3#w'][:], vgg16['conv3_3#b'][:], vgg16['conv4_1#w'][:], vgg16['conv4_1#b'][:],
            vgg16['conv4_2#w'][:], vgg16['conv4_2#b'][:], vgg16['conv4_3#w'][:], vgg16['conv4_3#b'][:],
            vgg16['conv5_1#w'][:], vgg16['conv5_1#b'][:], vgg16['conv5_2#w'][:], vgg16['conv5_2#b'][:],
            vgg16['conv5_3#w'][:], vgg16['conv5_3#b'][:]]]
        if with_fc:
            lasagne_model += [vgg16['fc6#w'][:], vgg16['fc6#b'][:],
                vgg16['fc7#w'][:], vgg16['fc7#b'][:]]
        lasagne.layers.set_all_param_values(net['pool5'], lasagne_model)
    print('VGG model built %s...' % (('with' if with_fc else 'without') + ' fully connect')) 
    return net

if __name__ == '__main__':
    net = make_vgg16(InputLayer((None, 3, 224, 224)), 'model/vgg16_weights_from_caffe.h5')
    paras = lasagne.layers.get_all_param_values(net['pool5'])
    # for para in paras:
    #     print para.shape
    # print net['pool5'].get_output_shape_for((10, 3, 224, 224))
    # print net['fc6'].get_output_shape_for((10, 3, 224, 224))
    # print net['fc7'].get_output_shape_for((10, 3, 224, 224))
    # print net['conv5_3'].get_output_shape_for((10, 3, 224, 224))
    for subnet in net:
        print(subnet, lasagne.layers.get_output_shape(net[subnet]))


