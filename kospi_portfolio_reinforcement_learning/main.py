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


#preprocessed data loading
is_train = 1

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 8
num_episodes = 10000 if is_train ==1 else 1
money = 1e+8

#saving
save_frequency = 100
save_path = './weights'
save_model = 1
load_model = 1
selecting_random = True
if is_train==0:
    env = environment.env(train = 0,number_of_asset = num_of_asset)
    load_model = 1
    selecting_random = False
else:
    env = environment.env(number_of_asset = num_of_asset)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

a_loss_sum = 0
s_loss_sum = 0

sess = tf.Session(config = config)

with tf.variable_scope('ESM'):
    selector = network.select_network(sess)
with tf.variable_scope('AAM'):
    allocator=network.policy(sess,num_of_asset = num_of_asset)

sess.run(tf.global_variables_initializer())
    

saver_ESM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ESM'))
saver_AAM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'AAM'))
ckpt_ESM = tf.train.get_checkpoint_state(save_path+'ESM')
ckpt_AAM = tf.train.get_checkpoint_state(save_path+'AAM')
if load_model:
    saver_ESM.restore(sess,ckpt_ESM.model_checkpoint_path)
    saver_AAM.restore(sess,ckpt_AAM.model_checkpoint_path)
        
score = 0
bench = 0
for i in range(num_episodes):
    allocator_memory = deque()
    selector_memory = deque()
    s=env.start()
    s=MM_scaler(s)
    done=False
    v=money
    weight_memory = []
    value_list = []
    while not done:
        evaluated_value = selector.predict(s)
        selected_s = env.selecting(evaluated_value,rand=selecting_random)
        selected_s = MM_scaler(selected_s)
        w = allocator.predict(selected_s)
        s_prime,r,done,v_prime,growth = env.action(w)
        s_prime=MM_scaler(s_prime)
        allocator_memory.append([selected_s,r])
        selector_memory.append([s,growth])
        weight_memory.append(w)
        s = s_prime
        v = v_prime
        value_list.append(v)
        if done:
            score+=v/money
            if is_train ==1:
                a_loss = allocator.update(allocator_memory)
                s_loss = selector.update(selector_memory)
                s_loss_sum += s_loss
                print(i,'agent:',round(v/money,4), 'benchmark:',round(env.benchmark/money,4))
                a_loss_sum += a_loss
            else:
                print(i,'agent:',round(v/money,4), 'benchmark:',round(env.benchmark/money,4))#, 'acc: ',round((env.acc[0,0]+env.acc[1,2])/(np.sum(env.acc)-np.sum(env.acc[:,1])),4))
               
    if save_model == 1 and i % save_frequency == save_frequency - 1:
        saver_ESM.save(sess,save_path+'ESM/ESM-'+str(i)+'.cptk')
        print('ESM loss:', s_loss_sum)
        s_loss_sum = 0
        saver_AAM.save(sess,save_path+'AAM/AAM-'+str(i)+'.cptk')
        print('average return: ',(round(score/save_frequency,4)-1)*100,'%')
        print('AAM loss:', a_loss_sum)
        print('saved')
        score = 0
        bench = 0
        a_loss_sum = 0
        
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