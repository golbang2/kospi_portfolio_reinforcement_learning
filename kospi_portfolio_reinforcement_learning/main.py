# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:43:58 2020

@author: golbang
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import scipy.signal
import network
import environment

#preprocessed data loading
path_data = './preprocess/price.npy'
asset_data = np.load(path_data, allow_pickle=True)

#hyperparameters
learning_rate = 3e-5
memory_size = 20
input_day_size = 50
filter_size = 3
num_of_feature = asset_data.shape[0]
num_of_asset = asset_data.shape[1]

#saving
save_frequency = 100
save_path = './algorithms'
save_model = 0
load_model = 0

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    Agent=network.policy(sess,num_of_feature,filter_size ,input_day_size, num_of_asset, memory_size,learning_rate = 1e-4, name='EIIE')
    Agent.sess.run(tf.global_variables_initializer())
    
    if save_model:
        saver = tf.train.Saver(max_to_keep=200)
        ckpt = tf.train.get_checkpoint_state(save_path)
        if load_model:
            saver.restore(sess,ckpt.model_checkpoint_path)
            
    