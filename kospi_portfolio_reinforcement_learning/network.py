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
        
        self._X=tf.placeholder(tf.float32,[None,self.output_size,self.input_size,self.num_of_feature],name="s")
        self._M=tf.placeholder(tf.float32,[None,self.output_size,1,self.memory_size],name='M')
        self._Y=tf.placeholder(tf.float32,[None,self.output_size+1],name='y')
        self._r=tf.placeholder(tf.float32, name='r')
        
        self.conv1 = layer.conv2d(self._X, 2, [1,self.filter_size], padding='VALID',activation_fn = tf.nn.relu)
        self.conv2 = layer.conv2d(self.conv1, 1, [1,self.input_size-self.filter_size+1], padding='VALID',activation_fn = tf.nn.relu)
        
        self.feature_map = tf.concat([self.conv2,self._M],axis=3)
        
        self.conv3 = layer.conv2d(self.feature_map, 1, [1,1], padding='VALID')
        
        self._B = tf.placeholder(tf.float32,[None,1,1,1],name='c')
        #self._B = tf.Variable([[[[1]]]],dtype=tf.float32)
        self.concat_cash = tf.concat([self.conv3, self._B],axis=1)
                
        self.policy=tf.nn.softmax(self.concat_cash,dim=1)
        
        self.log_p = self._Y * tf.log(tf.clip_by_value(self.policy,1e-10,1.))
        self.loss = -tf.reduce_mean(tf.reduce_sum(self.log_p, axis=1))
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self, s, memory):
        s = np.expand_dims(s,axis=0)
        memory = np.expand_dims(memory,axis=0)
        c = np.ones((1,1,1,1))
        self.weight=self.sess.run(self.policy, {self._X: s ,self._M: memory, self._B: c})
        return self.weight
        
    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        w = np.array(episode_memory[:,1].tolist())[:,0,:,0,0]
        r = np.array(episode_memory[:,2].tolist())        
        discounted_r = self.discounting_reward(r)
        memory = np.array(episode_memory[:,3].tolist())
        c = np.ones((len(episode_memory),1,1,1),dtype=np.float32)
        self.sess.run([self.loss,self.train], {self._X: s, self._Y: w, self._r: discounted_r, self._M: memory,self._B: c})
    
    def discounting_reward(self,r):
        discounted_r = np.zeros_like(r,dtype = np.float32)
        running_add = 0
        for t in reversed(range(len(r))):
            running_add = running_add *0.98 +r[t]
            discounted_r[t] = running_add
        return discounted_r
        
        