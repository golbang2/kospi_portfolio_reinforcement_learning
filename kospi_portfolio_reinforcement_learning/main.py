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

def memory_queue(memory,weight):
    memory = np.concatenate((w[0,1:],memory),axis=2)[:,:,:-1]
    return memory

def normalize_tensor(s):
    v_tensor = deque()
    for j in range(num_of_asset):
        v_array = deque()
        for i in range(input_day_size):
            v_array.append(s[j,i]/s[j,-1])
        v_array = np.array(v_array)
        v_tensor.append(v_array)
    v_tensor = np.array(v_tensor)
    return v_tensor

    
#preprocessed data loading
path_data = './preprocess/price_tensor.npy'
asset_data = np.load(path_data, allow_pickle=True)

#hyperparameters
learning_rate = 3e-5
memory_size = 20
input_day_size = 50
filter_size = 3
num_of_feature = asset_data.shape[2]
num_of_asset = asset_data.shape[0]
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
        s=normalize_tensor(s)
        done=False
        m = np.zeros([10,1,20],dtype=np.float32)
        print(i)
        while not done:
            w = agent.predict(s,m)
            s_prime,r,done = env.action(w)
            s_prime=normalize_tensor(s_prime)
            episode_memory.append([s,w,r,m])
            memory_queue(m,w)
            s = s_prime
            if done:
                agent.update(episode_memory)
                
        if i%save_frequency == 0 and i!=0:
            saver.save(sess,save_path+'/model-'+str(i)+'.cptk')
            print('saved')