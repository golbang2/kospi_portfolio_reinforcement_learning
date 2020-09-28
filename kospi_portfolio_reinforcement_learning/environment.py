# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:52:48 2020

@author: golbang
"""

import numpy as np
import pandas as pd
from collections import deque
import sys

class load_data:
    def __init__(self,data_path = './data/',day_length=50, train_length = 1000, test_length = 200):
        #hyper parameter
        self.number_of_asset = 10
        self.number_of_feature = 4 # close,high,low,volume
        self.train_length = train_length
        self.test_length = test_length
        self.day_length = day_length
        self.index_deque = deque()
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
    
    def value_index(self,index,value):
        self.value_deque.append([index,value])
        
    def sort_index(self):
        index_array = np.array(self.value_deque,dtype = np.float32)
        self.selected_index = []
        for i in range(number_of_asset):
            max_value_index = index_array[0,np.argmax(index_array[:,1])]
            self.selected_index.append(max_value_index)
            #delete
        self.value_deque = deque()
        return self.selected_index
        

class env:
    def __init__(self, path_data='./preprocess/', day_length=50,train=True, decimal = False):
        self.env_data = load_data()
        self.train = train
        self.day_len = day_length
        self.decimal = decimal
    
    def start(self,money=10000000):
        
        self.time = 0
        self.done = False
        self.state = self.env_array[:,self.time : self.time + self.day_len,:]
        self.value = money
        self.venchmark = np.sum(self.env_array[:,-1,0]/self.env_array[:,0,0])/10
        
        return self.state
    
    def action(self,weight):
        self.state = self.env_array[:, self.time : self.time + self.day_len,:]
        self.time = self.time + 1
        self.state_prime = self.env_array[:, self.time : self.time + self.day_len,:]
        self.y = self.state_prime[:,-1,0]/self.state[:,-1,0]
        self.calculate_value(weight)
        self.r = np.expand_dims((self.y-1)*100,axis=0)
        if self.time == self.env_array.shape[1]-self.day_len:
            self.done = True
        return self.state_prime, self.r, self.done, self.value

    def calculate_value(self,weight):
        if self.decimal == True:
            self.value = self.value*np.sum(weight*self.y)
        else:
            self.portfolio = (weight*self.value//self.state[:,-1,0])
            self.value = np.sum(self.portfolio * self.state_prime[:,-1,0]) + np.sum(weight*self.value % self.state[:,-1,0])
        
       

w = np.array([0.2,0.1,0.05,0.1,0.05,0.3,0.03,0.02,0.05,0.1])