# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:20:16 2020

@author: golbang
"""
import tensorflow as tf
import tensorflow.contrib.layers as layer
import numpy as np

class policy:
    def __init__(self, sess, num_of_feature, day_length, num_of_asset, memory_size=20 , filter_size = 3, learning_rate = 1e-4, name='allocator'):
        self.sess = sess
        self.input_size=day_length
        self.output_size=num_of_asset
        self.net_name=name
        self.memory_size = memory_size
        self.num_of_feature = num_of_feature
        self.filter_size = filter_size
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        initializer = layer.xavier_initializer()
        
        self._X=tf.placeholder(tf.float32,[None,self.output_size,self.input_size,self.num_of_feature],name="s") # shape: batch,10,50,4
        self._M=tf.placeholder(tf.float32,[None,self.output_size,1,self.memory_size],name='M')
        self._r=tf.placeholder(tf.float32,[None,self.output_size],name='r')
        
        self.conv1 = layer.conv2d(self._X, 8, [1,self.filter_size], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        self.conv2 = layer.conv2d(self.conv1, 1, [1,self.input_size-self.filter_size+1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer, weights_regularizer = regularizer)
        
        self.feature_map = tf.concat([self.conv2,self._M],axis=3)
        
        self.conv3 = layer.conv2d(self.feature_map, 1, [1,1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer, weights_regularizer = regularizer)
        
        self.conv3 = (self.conv3 - tf.reduce_min(self.conv3))/(tf.reduce_max(self.conv3)-tf.reduce_min(self.conv3))
        self.fc1 = layer.fully_connected(layer.flatten(self.conv3), 50, activation_fn=tf.nn.leaky_relu ,weights_regularizer = regularizer)
        self.fc2 = layer.fully_connected(self.fc1,20,activation_fn = tf.nn.leaky_relu,weights_regularizer = regularizer)
        self.policy = layer.fully_connected(self.fc2, self.output_size, activation_fn=tf.nn.softmax)
        
        self.loss = -tf.reduce_sum((self._r)*tf.log(self.policy+1e-6))
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self, s, memory):
        s = np.expand_dims(s,axis=0)
        memory = np.expand_dims(memory,axis=0)
        self.weight=self.sess.run(self.policy, {self._X: s ,self._M: memory})
        return self.weight
        
    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        r = np.array(episode_memory[:,1].tolist())[:,0]        
        memory = np.array(episode_memory[:,2].tolist())
        v = np.array(episode_memory[:,3].tolist())
        r = r-v
        self.sess.run([self.loss,self.train], {self._X: s, self._r: r, self._M: memory})
    
    def discounting_reward(self,r):
        discounted_r = np.zeros_like(r,dtype = np.float32)
        running_add = 0
        for t in reversed(range(len(r))):
            running_add = running_add *0.99 +r[t]
            discounted_r[t] = running_add
        return discounted_r
        
class select_network:
    def __init__(self, sess, num_of_feature = 4, filter_size = 3, day_length = 50, learning_rate = 1e-4, name = 'selector'):
        self.sess = sess
        self.net_name = name
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        initializer = layer.xavier_initializer()
        
        self._X = tf.placeholder(tf.float32,[None, day_length, num_of_feature,1]) # shape: batch,50,4,1
        self._y = tf.placeholder(tf.float32,[None])
        
        self.conv1 = layer.conv2d(self._X, 2, [filter_size, num_of_feature], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        self.fc1 = layer.fully_connected(layer.flatten(self.conv1), 32, activation_fn = tf.nn.leaky_relu, weights_regularizer = regularizer)
        self.v = layer.fully_connected(self.fc1,1)
        
        self.loss = tf.reduce_sum(tf.square(self._y - self.v))
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self,s):
        s = np.expand_dims(s,axis = -1)
        self.pi_hat = self.sess.run(self.v,{self._X:s})
        return self.pi_hat
        
    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        pi = np.array(episode_memory[:,1].tolist())
        for i in range(len(episode_memory)):
            ss = np.expand_dims(s[i],axis=-1)
            v = pi[i][0]
            self.sess.run([self.loss,self.train], {self._X: ss, self._y: v})