# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:52:48 2020

@author: golbang
"""

import numpy as np
import pandas as pd
from collections import deque

class load_data:
    def __init__(self,data_path = './data/',day_length=50, train_length = 1000, test_length = 200, train = True):
        #hyper parameter
        self.number_of_asset = 10
        self.number_of_feature = 4 # close,high,low,volume
        self.train_length = train_length
        self.test_length = test_length
        self.day_length = day_length
        self.index_deque = deque()
        self.value_deque = deque()
        self.max_len = 0
        a=0
        
        #load kospi200 list
        read_csv_file = 'KOSPI200.csv'
        ksp_list = np.loadtxt('./data/'+read_csv_file, delimiter=',',dtype = str)
        self.loaded_list = []
        ksp_list = ksp_list[:,0]
        
        for i in ksp_list:
            ksp_data = pd.read_csv("./data/stock_price/"+i+".csv")
            ksp_data = ksp_data[['Close','High','Low','Volume']].to_numpy(dtype=np.float32)
            if train:
                ksp_data = ksp_data[:-test_length]
            else:
                ksp_data = ksp_data[-test_length-day_length:]
            self.loaded_list.append(ksp_data)
            self.index_deque.append([a,i,len(ksp_data)])
            if self.max_len<len(ksp_data):
                self.max_len = len(ksp_data)
            a+=1

    def extract_selected(self,index,time):
        extract_deque = deque()
        for i in index:
            extract_deque.append(self.loaded_list[i][-(self.max_len-time):-(self.max_len-time-self.day_length),:])
        return np.array(extract_deque,dtype = np.float32)

    def sampling_data(self, time):
        sample_list = []
        for i in self.index_deque:
            if i[2]-self.day_length>=self.max_len-self.day_length-time:
                sample_list.append(i[0])
        return sample_list

    def extract_close(self,index,time):
        extract_deque = deque()
        for i in index:
            extract_deque.append(self.loaded_list[i][-(self.max_len-self.day_length-time),0])
        return np.array(extract_deque,dtype = np.float32)

class env:
    def __init__(self, decimal = False, day_length = 50, number_of_asset = 10, train = 1):
        self.train = train
        self.env_data = load_data(train = train)
        self.decimal = decimal
        self.number_of_asset = number_of_asset
        self.day_length = day_length
        
    def start(self,money=1e+8):
        self.time = 0
        self.done = False
        self.all_index = self.env_data.sampling_data(self.time)
        self.state = self.env_data.extract_selected(self.all_index,self.time)
        self.value = money
        self.all_memory = self.initialize_value_memory()
        
        self.acc = np.zeros([2,3],dtype=np.int32)
        return self.state,self.all_memory[tuple([self.all_index])]

    def selecting(self,value_array):
        self.selected_index = []
        self.sorted_value = value_array[:,0].argsort()[::-1][:self.number_of_asset]
        for i in self.sorted_value:
            self.selected_index.append(self.all_index[i])
        self.selected_state = self.env_data.extract_selected(self.selected_index,self.time)
        self.queue_value_memory(value_array)
        self.selected_memory = self.selecting_memory()
        self.selected_memory = np.expand_dims(self.selected_memory,1)
        
        self.value_array = value_array
        
        return self.selected_state, self.selected_memory

    def action(self,weight):
        self.r = self.calculate_value(weight)
        self.time += 1
        self.individual_return = self.calculate_individual_return()
        self.all_index = self.env_data.sampling_data(self.time)
        self.state_prime = self.env_data.extract_selected(self.all_index,self.time)
        self.r = np.expand_dims(self.r*100,axis=0)
        if self.time == self.env_data.max_len-self.day_length-1:
            self.done = True

        for i in range(len(self.individual_return)):
            if self.individual_return[i]>0 and self.value_array[i]>0:
                self.acc[0,0]+=1
            if self.individual_return[i]==0 and self.value_array[i]>0:
                self.acc[0,1]+=1
            if self.individual_return[i]<0 and self.value_array[i]>0:
                self.acc[0,2]+=1
            if self.individual_return[i]>0 and self.value_array[i]<0:
                self.acc[1,0]+=1
            if self.individual_return[i]==0 and self.value_array[i]<0:
                self.acc[1,1]+=1
            if self.individual_return[i]<0 and self.value_array[i]<0:
                self.acc[1,2]+=1

        return self.state_prime, self.r, self.done, self.value , self.individual_return ,self.all_memory[tuple([self.all_index])]

    def calculate_value(self,weight):
        close = self.env_data.extract_close(self.selected_index,self.time-1)
        close_prime = self.env_data.extract_close(self.selected_index,self.time)
        self.y = (weight*self.value//close)
        self.value = np.sum(self.y * close_prime) + np.sum(weight * self.value % close)
        return np.log(close_prime/close)

    def calculate_individual_return(self):
        close = self.env_data.extract_close(self.all_index,self.time-1)
        close_prime = self.env_data.extract_close(self.all_index,self.time)
        return np.log(close_prime/close)

    def initialize_value_memory(self):
        self.memory_deque=deque()
        for i in self.env_data.index_deque:
            self.memory_deque.append(np.zeros([20],dtype=np.float32))
        return np.array(self.memory_deque,dtype=np.float32)

    def queue_value_memory(self,value):
        for i in range(len(self.all_index)):
            self.memory_deque[self.all_index[i]] = np.append(self.memory_deque[self.all_index[i]][1:],value[i])

    def selecting_memory(self):
        self.selected_memory_deque = deque()
        for i in self.selected_index:
            self.selected_memory_deque.append(self.memory_deque[i])
        return np.array(self.selected_memory_deque,dtype=np.float32)