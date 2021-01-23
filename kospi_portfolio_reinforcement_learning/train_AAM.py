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

def validation(v_env,AAM):
    v_env.start()
    v_d = False
    v_r_s = 0
    while not v_d:
        v_selected_s = v_env.selecting_rand()
        v_selected_s = MM_scaler(v_selected_s)
        v_w = AAM.predict(v_selected_s)
        _,v_r,v_d,_,_ = env.action(w)
        v_r_s += np.sum(v_r*v_w)
    return v_r_s
        
def repeat_episode(r_env,AAM,repeat = 100):
    v_list = []
    for i in range(repeat):
        v_list.append(validation(r_env,AAM))
    return v_list

#preprocessed data loading
is_train = 1

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 8
num_episodes = 5000
money = 1e+8

#saving
save_frequency = 100
save_path = './weights/AAM/m_'
save_model = 1
load_model = 0
selecting_random = True

env = environment.env(number_of_asset = num_of_asset)
env_val = environment.env(number_of_asset = num_of_asset,train = 0,validation = 1)


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

a_loss_sum = 0

sess = tf.Session(config = config)

with tf.variable_scope('AAM'):
    allocator=network.policy(sess,num_of_asset = num_of_asset)
#with tf.variable_scope('ESM'):    
#    selector = network.select_network(sess)

sess.run(tf.global_variables_initializer())

#saver_ESM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ESM'),max_to_keep=100)
#ckpt_ESM = tf.train.get_checkpoint_state('./weights/ESM')
saver_AAM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'AAM'),max_to_keep=100)
ckpt_AAM = tf.train.get_checkpoint_state(save_path+str(num_of_asset))

if load_model:
    saver_AAM.restore(sess,ckpt_AAM.model_checkpoint_path)
    #saver_ESM.restore(sess,ckpt_ESM.model_checkpoint_path)

score = 0
bench = 0
best_val_perf = 0.

for i in range(num_episodes):
    allocator_memory = deque()
    s = env.start()
    done = False
    v = money
    value_list = []
    while not done:
        selected_s = env.selecting_rand()
        selected_s = MM_scaler(selected_s)
        w = allocator.predict(selected_s)
        s,r,done,v,growth = env.action(w)
        allocator_memory.append([selected_s,r])
        value_list.append(v)
    
    score += v/money
    a_loss = allocator.update(allocator_memory)
    print(i,'agent:',round(v/money,4), 'benchmark:',round(env.benchmark/money,4))
    a_loss_sum += a_loss

    if save_model == 1 and i % save_frequency == save_frequency - 1:
        print('average return: ',(round(score/save_frequency,4)-1)*100,'%')
        saver_AAM.save(sess,save_path+str(num_of_asset)+'/AAM-'+str(i)+'.cptk')
        print('AAM loss:', a_loss_sum)
        score = 0
        bench = 0
        a_loss_sum = 0
    