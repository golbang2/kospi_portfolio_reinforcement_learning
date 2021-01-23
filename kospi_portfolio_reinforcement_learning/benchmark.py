# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:05:26 2020

@author: golbang
"""


import environment
from collections import deque
import numpy as np

class benchmark:
    def __init__(self,num_of_asset=10):
        self.num_of_asset = num_of_asset
        
        self.env = environment.env(train=0,number_of_asset = num_of_asset)
        self.w = np.ones(self.num_of_asset,np.float32)/self.num_of_asset
        
        self.UCRP_deque = deque()
        self.UBAH_deque = deque()

    def UCRP(self):
        
        self.env.start()
        value_list = []
        done = False
        selected_index = np.random.choice(self.env.all_index,self.num_of_asset, False)
        while not done:
            self.env.holding(selected_index)
            _,_,done,v_prime,_ =self.env.action(self.w)
            value_list.append(v_prime)
        self.UCRP_deque.append(value_list)

    def UBAH(self):
        self.s = self.env.start()
        value_list = []
        done = False
        selected_index = np.random.choice(self.env.all_index,self.num_of_asset,False)
        _,_ = self.env.start_UBAH(selected_index, self.w)
        while not done:
            done = self.env.action_UBAH(selected_index, self.w)
            value_list.append(self.env.value)
        self.UBAH_deque.append(value_list)
    
    def execute(self,iteration = 100):
        for i in range(iteration):
            self.UCRP()
            self.UBAH()
        return np.array(self.UCRP_deque),np.array(self.UBAH_deque)
    
    def calculate_st(self):
        UCRP,UBAH = self.execute()
        UCRP_mean = np.mean(UCRP,axis = 0)
        UBAH_mean = np.mean(UBAH,axis = 0)
        
        return UCRP_mean, UBAH_mean