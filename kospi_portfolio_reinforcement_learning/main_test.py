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

def calculate_SV(value_list):
    value_array = np.array(value_list)/money
    return_array = np.log(value_array[1:]/value_array[:-1])
    SV=np.sqrt(np.var(return_array))
    SV = SV*np.sqrt(period)
    return SV

def sharpe_ratio(value_list,kospi):
    value_array = np.array(value_list)
    sigma = calculate_SV(value_list)
    APV = value_array[-1]
    excess_return = APV-kospi[-1]/kospi[0]
    sharpe_ratio = excess_return/sigma
    return sharpe_ratio

def MDD(apv):
    apv_max=[]
    apv_min=[]
    for i in range(1,period):
        apv_max.append(np.max(apv[:i]))
        apv_min.append(np.min(apv[np.argmax(apv[:i]):i]))
    drawdown = (np.array(apv_max) - np.array(apv_min))/np.array(apv_max)
    return drawdown

#preprocessed data loading
is_train = 0

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 10
num_episodes = 10000 if is_train ==1 else 1
test_episodes = 2000
money = 1e+8
period = 1019 if is_train else 255

#saving
save_frequency = 100
save_path = './weights/'
save_model = 1
load_model = 1
selecting_random = True
if is_train==0:
    env = environment.env(train = 0, number_of_asset = num_of_asset)
    load_model = 1
    selecting_random = False
else:
    env = environment.env(number_of_asset = num_of_asset)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

a_loss_sum = 0
s_loss_sum = 0
value_deque = deque()
AAM_deque = deque()

sess = tf.Session(config = config)

with tf.variable_scope('AAM'):
    allocator=network.policy(sess,num_of_asset = num_of_asset)
with tf.variable_scope('ESM'):    
    selector = network.select_network(sess)

sess.run(tf.global_variables_initializer())

saver_AAM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'AAM'),max_to_keep=100)
ckpt_AAM = tf.train.get_checkpoint_state(save_path+'AAM/m_'+str(num_of_asset))
saver_ESM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ESM'),max_to_keep=100)
ckpt_ESM = tf.train.get_checkpoint_state(save_path+'ESM')

if load_model:
    saver_AAM.restore(sess,ckpt_AAM.model_checkpoint_path)
    saver_ESM.restore(sess,ckpt_ESM.model_checkpoint_path)

score = 0
for i in range(num_episodes):
    allocator_memory = deque()
    selector_memory = deque()
    s=env.start()
    s=MM_scaler(s)
    done=False
    v=money
    weight_memory = []
    selected_memory = []
    value_list = []
    bench_list = []
    while not done:
        evaluated_value = selector.predict(s)
        selected_s = env.selecting(evaluated_value)
        selected_memory.append(env.selected_index)
        selected_s = MM_scaler(selected_s)
        w = allocator.predict(selected_s)
        s_prime,r,done,v_prime,growth = env.action(w)
        s_prime=MM_scaler(s_prime)
        weight_memory.append(w)
        s = s_prime
        v = v_prime
        value_list.append(v)
        bench_list.append(env.benchmark)
        if done:
            value_deque.append(['AAM+ESM',value_list])
            value_deque.append(['ESM only',bench_list])
            score+=v/money
            print(i,'agent:',round(v/money,4), 'benchmark:',round(env.benchmark/money,4))


'''
for i in range(test_episodes):
    value_list = []
    allocator_memory = deque()
    s=env.start()
    s = MM_scaler(s)
    done = False
    v = money
    while not done:
        selected_s = env.selecting(evaluated_value, rand = True)
        selected_s = MM_scaler(selected_s)
        w = allocator.predict(selected_s)
        s_prime,r,done,v_prime,growth = env.action(w)
        s_prime = MM_scaler(s_prime)
        s = s_prime
        v = v_prime
        value_list.append(v)
        if done:
            AAM_deque.append(value_list)
            score+=v/money
            print(i,'agent:',round(v/money,4), 'benchmark:',round(env.benchmark/money,4))

import benchmark
import pandas as pd

value_array = np.array(value_deque[0][1])
ESM_value_array = np.array(value_deque[1][1])
AAM_mean_deque = deque()
AAM_array = np.array(AAM_deque)
for i in range(env.time):
    AAM_mean_deque.append(np.mean(AAM_array[:,i]))
AAM_mean_array = np.array(AAM_mean_deque)
AAM_mean_array

k200price = pd.read_csv("./data/KOSPI200index.csv")
k200 = k200price[['종가']].to_numpy(dtype=np.float32)
k200 = k200[1:-1]
k200 = k200[::-1]

benchmark = benchmark.benchmark(num_of_asset = num_of_asset)
ucrp,ubah = benchmark.execute(iteration=test_episodes)
UCRP = np.mean(ucrp,axis = 0)
UBAH = np.mean(ubah,axis = 0)

best = env.env_data.loaded_list[86][50:-1,0]
best = best/env.env_data.loaded_list[86][49,0]

base_rate = np.array(np.loadtxt('./data/base_rate_test.csv',delimiter=',',dtype = str)[1:][:,1],dtype =np.float32)

value_array= value_array[:-19]
ESM_value_array = ESM_value_array[:-19]
AAM_mean_array = AAM_mean_array[:-19]
k200 = k200[:-19]
UBAH = UBAH[:-19]
UCRP = UCRP[:-19]
best = best[:-19]

plt.figure(figsize = (10,4))
plt.grid(1)
plt.plot(value_array/money,label='ESM+AAM')
plt.plot(ESM_value_array/money,label = 'ESM')
plt.plot(AAM_mean_array/money,label = 'AAM')
plt.plot(k200/k200[0],label='KOSPI200')
plt.plot(UBAH/money,label = 'UBAH')
plt.plot(UCRP/money,label = 'UCRP')
plt.plot(best,label = 'Best')
plt.xlabel("time")
plt.ylabel("APV")
plt.legend()
plt.show()
'''

select_weight = np.zeros((200,period),dtype=np.float)
for i in range(period):
    for j in range(num_of_asset):
        select_weight[selected_memory[i][j],i]=weight_memory[i][0,j]

index = 0
for i in range(5):
    fig, axs = plt.subplots(nrows=5, ncols=8, figsize=(16,5),constrained_layout=True)
    for ax in axs.flat:
        ax.plot(select_weight[index])
        ax.set_yscale('log',basey=10)
        ax.xaxis.set_visible(False)
        ax.set_title(label=env.env_data.index_deque[index][1])
        index+=1
    plt.show()
    

for i in range(6):
    plt.figure(figsize=(10,1.8))
    plt.grid(1,axis='y')
    plt.bar(index,weight_memory[i*50][0])
    plt.xticks(index, index_deque[i*50])
    plt.xlabel('code')
    plt.ylabel('weight')
    plt.title('Weight at time t='+str(i*50))
    plt.savefig('fig'+str(i)+'.png')    
    plt.show()
