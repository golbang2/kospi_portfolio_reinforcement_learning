# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 00:49:15 2020

@author: golbang
"""


import tensorflow as tf
import tensorflow.contrib.layers as layer
import tensorflow.nn as nn
import numpy as np
import environment

num_of_feature = 10
filter_size=3
memory_size = 20
num_of_asset= 10

X=tf.placeholder(tf.float32,[None,4,10,50],name="s")
M=tf.placeholder(tf.float32,[None,20,10,1],name='M')

conv1 = layer.convolution2d(activation_fn=tf.nn.relu, inputs=X, num_outputs=2,kernel_size=[3,1], stride=1)

envi=environment.env()
s=envi.start()
s = np.expand_dims(s,axis=0)

memory = np.zeros([memory_size,num_of_asset,1],dtype = np.float32)
memory = np.expand_dims(memory,axis=0)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(conv1, {X: s, M: memory})