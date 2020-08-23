# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:52:48 2020

@author: golbang
"""


import numpy as np

class env:
    def __init__(self, path_data='./preprocess/price.npy', day_length=50):
        self.day_len = day_length
        self.env_array = np.load(path_data, allow_pickle=True)
        self.time = 0
        self.done = False
        
    def action(self,weight):
        self.state = self.env_array[:, :, self.time : self.time + self.day_len]
        self.time = self.time + 1
        self.state_prime = self.env_array[:, :, self.time : self.time + self.day_len]
        self.y = self.state_prime[0,:,-1]/self.state[0,:,-1]
        self.r = np.sum(np.log(self.y)*weight)
        if self.time == self.state.shape[2]-self.day_len:
            self.done = True
        return self.state_prime, self.r, self.done
    
    
w = np.array([0.2,0.1,0.05,0.1,0.05,0.3,0.03,0.02,0.05,0.1])