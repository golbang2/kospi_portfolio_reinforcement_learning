# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:20:16 2020

@author: golbang
"""
import tensorflow as tf
import tensorflow.contrib.layers as layer
import numpy as np

class policy:
    def __init__(self, sess, num_of_feature=4, day_length=50, num_of_asset=10,  filter_size = 3, learning_rate = 1e-4, regularizer_rate = 0.1, name='allocator'):
        self.sess = sess
        self.input_size=day_length
        self.output_size=num_of_asset
        self.net_name=name
        self.num_of_feature = num_of_feature
        self.filter_size = filter_size
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        initializer = layer.xavier_initializer()
        
        self._X=tf.placeholder(tf.float32,[None,self.output_size,self.input_size,self.num_of_feature],name="s") # shape: batch,10,50,4
        self._r=tf.placeholder(tf.float32,[None,self.output_size],name='r')
        
        self.conv1 = layer.conv2d(self._X, 8, [1,self.filter_size], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        self.conv2 = layer.conv2d(self.conv1, 1, [1,self.input_size-self.filter_size+1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer, weights_regularizer = regularizer)
        
        self.fc1 = layer.fully_connected(layer.flatten(self.conv2), 50, activation_fn=tf.nn.leaky_relu, weights_regularizer = regularizer)

        self.policy = layer.fully_connected(self.fc1, self.output_size, activation_fn=tf.nn.softmax)
        
        self.loss = -tf.reduce_sum((self._r)*(self.policy))
        
        self.train = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self, s):
        s = np.expand_dims(s,axis=0)
        self.weight=self.sess.run(self.policy, {self._X: s})
        return self.weight

    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        r = np.array(episode_memory[:,1].tolist())[:,0]
        a_loss = self.sess.run([self.loss,self.train], {self._X: s, self._r: r})[0]
        return a_loss
        
class select_network:
    def __init__(self, sess, num_of_feature = 4, filter_size = 3, day_length = 50, learning_rate = 1e-4, name = 'selector'):
        self.sess = sess
        self.net_name = name
        
        self._X = tf.placeholder(tf.float32,[None, day_length, num_of_feature]) # shape: batch,50,4
        self._y = tf.placeholder(tf.float32,[None])
        
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units = 5)
        self.multicell = tf.contrib.rnn.MultiRNNCell([self.cell]*2)
        self.lstm1, _states = tf.nn.dynamic_rnn(self.cell, self._X, dtype=tf.float32)
        self.value = layer.fully_connected(self.lstm1[:,-1], 1, activation_fn=None)
        
        self.loss = tf.reduce_sum(tf.square(self._y - self.value))
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self,s):
        self.value_hat = self.sess.run(self.value,{self._X:s})
        return self.value_hat
        
    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        v = np.array(episode_memory[:,1].tolist())
        loss = 0
        for i in range(len(episode_memory)):
            loss+= self.sess.run([self.loss,self.train], {self._X: s[i], self._y: v[i]})[0]
        #print('ESM loss :',loss)
        return loss