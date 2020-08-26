# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:43:58 2020

@author: golbang
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
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
num_episodes = 20

#saving
save_frequency = 100
save_path = './algorithms'
save_model = 0
load_model = 0
env = environment.env()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    agent=network.policy(sess,num_of_feature,input_day_size,num_of_asset,memory_size,filter_size,learning_rate = 1e-4,name='EIIE')
    agent.sess.run(tf.global_variables_initializer())
    
    if save_model:
        saver = tf.train.Saver(max_to_keep=200)
        ckpt = tf.train.get_checkpoint_state(save_path)
        if load_model:
            saver.restore(sess,ckpt.model_checkpoint_path)
    
    for i in range(num_episodes):
        episode_memory = deque()
        s=env.start()
        done=False
        score=0
        m = np.zeros([num_of_asset,1,memory_size], dtype = np.float32)
        while not done:
            w = agent.predict(s)
            s_prime,r,done = env.action(w)
            episode_memory.append([s,w,r,m])
            s = s_prime
            if done:
                episode_memory = np.array(episode_memory)
                agent.update(episode_memory)
                
        if i%save_frequency == 0:
            saver.save(sess,save_path+'/model-'+str(i)+'.cptk')
            print('saved')