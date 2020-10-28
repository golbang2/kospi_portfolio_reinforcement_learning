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
is_train = 1

#hyperparameters
learning_rate = 3e-5
memory_size = 20
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 10
num_episodes = 2000 if is_train ==1 else 1
money = 1e+8

#saving
save_frequency = 100
save_path = './algorithms'
save_model = 1
load_model = 1
if is_train==0:
    env = environment.env(train = 0)
    load_model = 1
else:
    env = environment.env()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    allocator=network.policy(sess)
    selector = network.select_network(sess)
    sess.run(tf.global_variables_initializer())
    
    if save_model:
        saver = tf.train.Saver(max_to_keep=100)
        ckpt = tf.train.get_checkpoint_state(save_path)
        if load_model:
            saver.restore(sess,ckpt.model_checkpoint_path)
    score = 0
    vench = 0
    value_list = []
    for i in range(num_episodes):
        allocator_memory = deque()
        selector_memory = deque()
        s,m=env.start()
        s=MM_scaler(s)
        done=False
        v=money
        weight_memory = []
        while not done:
            evaluated_value= selector.predict(s,m)
            selected_s,selected_m = env.selecting(evaluated_value)
            selected_s = MM_scaler(selected_s)
            w= allocator.predict(selected_s,selected_m)
            s_prime,r,done,v_prime,growth,m_prime = env.action(w)
            s_prime=MM_scaler(s_prime)
            allocator_memory.append([selected_s,r,selected_m,np.repeat(((v_prime/v-1)*100),10)])
            selector_memory.append([s,(growth*10),m])
            weight_memory.append(w)
            s = s_prime
            v = v_prime
            m = m_prime
            value_list.append(v)
            if done:
                score+=v/money
                print(i,'agent:',round(v/money,4), 'acc: ',round((env.acc[0,0]+env.acc[1,2])/(np.sum(env.acc)-11452),4))
                if is_train ==1:
                    allocator.update(allocator_memory)
                    selector.update(selector_memory)

        if save_model == 1 and i % save_frequency == save_frequency - 1:
            saver.save(sess,save_path+'/model-'+str(i)+'.cptk')
            print('saved')
            print('average return: ',round_down(score/save_frequency),'%')
            score = 0
            vench = 0

'''
import pandas as pd
k200price = pd.read_csv("d:/kospi200price.csv")
k200 = k200price[['종가']].to_numpy(dtype=np.float32)
k200 = k200[1:-1]
k200 = k200[::-1]
v_array = np.array(value_list,dtype=np.float32)
plt.plot(v_array/v_array[0],label='agent')
plt.plot(k200/k200[0],label='ks200')
plt.xlabel("Time Period")
plt.ylabel("value")
plt.legend()
plt.show()
'''