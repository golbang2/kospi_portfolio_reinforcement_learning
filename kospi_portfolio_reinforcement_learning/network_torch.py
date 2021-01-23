# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:16:12 2020

@author: golbang
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class policy(nn.Module):
    def __init__(self, num_of_feature=4, day_length=50, num_of_asset=10, memory_size=20 , filter_size = 3, learning_rate = 1e-4, regularizer_rate = 0.1, name='allocator'):
        super(policy, self).__init__()
        
        self.input_size=day_length
        self.output_size=num_of_asset
        self.net_name=name
        self.memory_size = memory_size
        self.num_of_feature = num_of_feature
        self.filter_size = filter_size
        
        self.conv1 = nn.Conv2d(self.num_of_feature,8,(1,self.filter_size))
        self.conv2 = nn.Conv2d(8,2,(1,self.input_size-self.filter_size+1))
        
        self.fc=nn.Linear(50,self.output_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay = regularizer_rate)
        
    def forward(self, s):
        x = F.leaky_relu(self.conv1(s))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(10)
        weight = F.softmax(self.fc(x),dim=0)
        return weight
        
    def update(self):               
        R = r + gamma * R
        loss = -torch.log(prob) * R
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class evaluator(nn.Module):
    def __init__(self, num_of_feature = 4, filter_size = 3, day_length = 50, memory_size = 20, learning_rate = 1e-4, name = 'selector'):
        super(evaluator,self).__init()
        
        