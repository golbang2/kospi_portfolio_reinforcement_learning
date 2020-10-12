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
        x[i]=(s[i]-np.min(s[i],axis=0))/((np.max(s[i],axis=0)-np.min(s[i],axis=0))+1e-5)
    return x

def round_down(x):
    return round((x-1)*100,4)
    

#preprocessed data loading
is_train = 0

#hyperparameters
learning_rate = 3e-5
memory_size = 20
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 10
num_episodes = 30000 if is_train ==1 else 1
money = 1e+8

#saving
save_frequency = 100
save_path = './algorithms'
save_model = 1
load_model = 0
if is_train==0:
    env = environment.env(train = 0)
    load_model = 1
else:
    env = environment.env()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    allocator=network.policy(sess,num_of_feature,input_day_size,num_of_asset,memory_size,filter_size,learning_rate = learning_rate,name='EIIE')
    selector = network.select_network(sess)
    sess.run(tf.global_variables_initializer())
    
    if save_model:
        saver = tf.train.Saver(max_to_keep=3)
        ckpt = tf.train.get_checkpoint_state(save_path)
        if load_model:
            saver.restore(sess,ckpt.model_checkpoint_path)
    score = 0
    vench = 0
    value_list = []
    for i in range(num_episodes):
        allocator_memory = deque()
        selector_memory = deque()
        s=env.start(money = money)
        s=MM_scaler(s)
        done=False
        m = np.ones([10,1,20],dtype=np.float32)
        v=money
        weight_memory = []
        while not done:
            conditional_pi = selector.predict(s)
            selected_s = env.selecting(conditional_pi)
            selected_s = MM_scaler(selected_s)
            w= allocator.predict(selected_s,m)
            s_prime,r,done,v_prime = env.action(w)
            s_prime=MM_scaler(s_prime)
            allocator_memory.append([selected_s,r,m,np.repeat(((v_prime/v-1)*100),10)])
            selector_memory.append([selected_s,r])
            weight_memory.append(w)
            m=memory_queue(m,w)
            s = s_prime
            v = v_prime
            value_list.append(v)
            if done:
                score+=v/money
                print(i,'agent:',v/money)
                if is_train ==1:
                    allocator.update(allocator_memory)
                    selector.update(selector_memory)

        if save_model == 1 and i % save_frequency == save_frequency - 1:
            saver.save(sess,save_path+'/model-'+str(i)+'.cptk')
            print('saved')
            print('average return: ',round_down(score/save_frequency),'%')
            print(score-vench)
            score = 0
            vench = 0

'''
plt.plot(v_array/v_array[0],label='agent')
plt.plot(k200/k200[0],label='ks200')
plt.xlabel("Time Period")
plt.ylabel("value")
plt.legend()
plt.show()
'''