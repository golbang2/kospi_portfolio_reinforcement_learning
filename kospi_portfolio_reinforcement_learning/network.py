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
    def __init__(self, sess, num_of_feature,filter_size ,day_length, num_of_asset, memory_size,learning_rate = 1e-4, name='EIIE'):
        self.sess = sess
        self.input_size=day_length
        self.output_size=num_of_asset
        self.net_name=name
        self.memory_size = memory_size
        self.num_of_feature = num_of_feature
        self.filter_size
        
        self._X=tf.placeholder(tf.float32,[None,self.num_of_feature,self.input_size,self.output_size,1],name="s")
        self._M=tf.placeholder(tf.float32,[None,self.memory_size,self.output_size,1],name='M')
        self._Y=tf.placeholder(tf.float32,[None,self.output_size],name='y')
        self._r=tf.placeholder(tf.float32, name='reward')
        
        self.conv1 = layer.convolution3d(activation_fn=tf.nn.relu, inputs=self._X, num_outputs=2, kernel_size=[4, filter_size, 1], stride = 1)
        self.conv2 = layer.convolution3d(activation_fn=tf.nn.relu, inputs=self.conv1, num_outputs=1, 
                                         kernel_size=[2, self.input_size-filter_size+1, 1]   , stride = 1)
        
        self.expanded_asset = tf.expand_dims(self.conv3,axis=-1)
        self.feature_map = tf.concat([self.expanded_asset,self._M],axis=0)
        
        self.cash_bias = tf.Variable([[1]],dtype=tf.float32)
        self.concat_cash = tf.concat([self.feature_map, self.cash_bias],axis=1)
        
        self.conv3 = layer.convolution3d(activation_fn=None, inputs = self.concat_cash, num_output=1,kernel_size = [memory_size+1, 1, 1], stride = 1)
        
        self.policy=tf.nn.softmax(self.policy_layer2)
        
        self.log_p = self._Y * tf.log(tf.clip_by_value(self.policy,1e-10,1.))
        self.loss = -tf.reduce_mean(tf.reduce_sum(self.log_p, axis=1))
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self, s, memory):
        self.tensor = self.normalize_tensor
        self.weight=self.sess.run(self.policy, {self._X: s,self._M: memory})
        return self.weight
        
    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = episode_memory[:,0]
        w = episode_memory[:,1]
        r = episode_memory[:,2]        
        discounted_r = self.discounting_reward(r)
        memory = episode_memory[:,3]
        self.sess.run([self.loss,self.train], {self._X: s, self._Y: w, self._r: discounted_r, self._M: memory})
        
    def normalize_tensor(self,s):
        self.v_tensor = deque()
        for j in range(self.num_of_feature):
            self.v_vector = deque()
            for i in range(self.input_size):
                self.v_vector.append(s[j,:,i]/s[j,:,-1])
            self.v_vector = np.array(self.v_vector)
            self.v_tensor.append(self.v_vector)
        self.v_tensor = np.array(self.v_tensor)
        return self.v_tensor
    
    def discounting_reward(self,r):
        discounted_r = np.zeors_like(r,dtype = np.float32)
        running_add = 0
        for t in reversed(range(len(r))):
            running_add = running_add *0.98 +r[t]
            discounted_r[t] = running_add
        return discounted_r
        