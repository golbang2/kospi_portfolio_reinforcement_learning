# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:38:07 2020

@author: golbang
"""


import network
import environment
import tensorflow as tf
import tensorflow.contrib.layers as layer
import numpy as np

env = environment.env()
s = env.start()
s = np.expand_dims(s,axis = -1)
memory = np.ones([184,20],dtype=np.float32)

X = tf.placeholder(tf.float32,[None, 50, 4, 1])
m = tf.placeholder(tf.float32,[None, 20])
y = tf.placeholder(tf.float32,[None])

initializer = layer.xavier_initializer()

conv1 = layer.conv2d(X, 1 , [3, 4], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
fc1 = layer.fully_connected(layer.flatten(conv1), 32, activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
fc2 = layer.fully_connected(fc1, 1, activation_fn=None)

concat_layer = tf.concat([m,fc2],axis=1)

concat_layer = tf.reshape(concat_layer, shape=[-1, 21, 1])

cell = tf.contrib.rnn.LSTMCell(num_units=1, state_is_tuple=True, activation=tf.tanh)
lstm1, _states = tf.nn.dynamic_rnn(cell, concat_layer, dtype=tf.float32)
value = layer.fully_connected(lstm1[:,-1], 1, activation_fn=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    v = sess.run(value,{X:s,m:memory})