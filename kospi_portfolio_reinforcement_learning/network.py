# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:20:16 2020

@author: golbang
"""
import tensorflow as tf
import tensorflow.contrib.layers as layer
import numpy as np
from collections import deque

class policy:
    def __init__(self, sess, num_of_feature, day_length, num_of_asset, memory_size=20 , filter_size = 3, learning_rate = 1e-4, name='EIIE'):
        self.sess = sess
        self.input_size=day_length
        self.output_size=num_of_asset
        self.net_name=name
        self.memory_size = memory_size
        self.num_of_feature = num_of_feature
        self.filter_size = filter_size
        self.state_deque = deque()
        self.memory_deque = deque()
        self.reward_deque = deque()
        
        self._X=tf.placeholder(tf.float32,[None,self.output_size,self.input_size,self.num_of_feature],name="s")
        self._M=tf.placeholder(tf.float32,[None,self.output_size,1,self.memory_size],name='M')
        self._r=tf.placeholder(tf.float32,[None,self.output_size],name='r')
        
        self.conv1 = layer.conv2d(self._X, 2, [1,self.filter_size], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = layer.xavier_initializer())
        self.conv2 = layer.conv2d(self.conv1, 1, [1,self.input_size-self.filter_size+1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = layer.xavier_initializer())
        
        self.feature_map = tf.concat([self.conv2,self._M],axis=3)
        
        self.conv3 = layer.conv2d(self.feature_map, 1, [1,1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = layer.xavier_initializer())
        
        self.scaled_conv3 = (self.conv3 - tf.reduce_min(self.conv3))/(tf.reduce_max(self.conv3)-tf.reduce_min(self.conv3))
        self.fc1 = layer.fully_connected(layer.flatten(self.scaled_conv3), 20, activation_fn=tf.nn.leaky_relu)
        self.policy = layer.fully_connected(self.fc1, self.output_size, activation_fn=tf.nn.softmax)
   
        self.loss = -tf.reduce_sum(self.policy*self._r)

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self, s, memory):
        s = np.expand_dims(s,axis=0)
        memory = np.expand_dims(memory,axis=0)
        self.weight=self.sess.run(self.policy, {self._X: s ,self._M: memory})
        return self.weight
        
    def update(self, episode_memory):
        self.state_deque = np.array(self.state_deque)
        self.memory_deque = np.array(self.memory_deque)
        self.reward_deque = np.array(self.reward_deque)
        self.sess.run([self.loss,self.train], {self._X: self.state_deque, self._r: self.reward_deque, self._M: self.memory_deque})
        self.state_deque = deque()
        self.memory_deque = deque()
        self.reward_deque = deque()
        
    def buffer(self,s,r,m):
        self.state_deque.append(s)
        self.memory_deque.append(m)
        self.reward_deque.append(r)
