# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:43:58 2020

@author: golbang
"""


import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import network
import environment
import numpy as np

def memory_queue(memory,weight):
    weight = np.expand_dims(weight.T,axis=1)
    memory = np.concatenate((weight,memory),axis=2)[:,:,:-1]
    return memory

def MM_scaler(s):
    x= np.zeros(s.shape)
    for i in range(len(s)):
        x[i]=(s[i]-np.min(s[i],axis=0))/(np.max(s[i],axis=0)-np.min(s[i],axis=0))
    return x
    

#preprocessed data loading
is_train = 'train'
path_data = './preprocess/'+is_train+'_price_tensor.npy'
asset_data = np.load(path_data, allow_pickle=True)

#hyperparameters
learning_rate = 3e-5
memory_size = 20
input_day_size = 50
filter_size = 3
num_of_feature = asset_data.shape[2]
num_of_asset = asset_data.shape[0]
num_episodes = 30000 if is_train =='train' else 1

#saving
save_frequency = 100
save_path = './algorithms'
save_model = 1
load_model = 0
if is_train=='test':
    env = environment.env(train = False)
else:
    env = environment.env()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    agent=network.policy(sess,num_of_feature,input_day_size,num_of_asset,memory_size,filter_size,learning_rate = learning_rate,name='EIIE')
    agent.sess.run(tf.global_variables_initializer())
    
    if save_model:
        saver = tf.train.Saver(max_to_keep=3)
        ckpt = tf.train.get_checkpoint_state(save_path)
        if load_model:
            saver.restore(sess,ckpt.model_checkpoint_path)
    
    for i in range(num_episodes):
        episode_memory = deque()
        s=env.start()
        s=MM_scaler(s)
        done=False
        m = np.ones([10,1,20],dtype=np.float32)
        while not done:
            w= agent.predict(s,m)
            s_prime,r,done,value = env.action(w)
            s_prime=MM_scaler(s_prime)
            agent.buffer(s,r,m)
            m=memory_queue(m,w)
            s = s_prime
            if done:
                print(i,value)
                if is_train =='train':
                    agent.update(episode_memory)
                
        if save_model == 1 and i % save_frequency == save_frequency - 1:
            saver.save(sess,save_path+'/model-'+str(i)+'.cptk')
            print('saved')