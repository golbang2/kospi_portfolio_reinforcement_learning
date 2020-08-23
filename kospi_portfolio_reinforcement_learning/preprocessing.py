# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:28:00 2020

@author: golbang
"""

number_of_asset = 10
number_of_feature = 4 # close,high,low,volume

import numpy as np
import pandas as pd
from collections import deque

ksp50 = np.loadtxt('./data/kospi60.csv', delimiter=',',dtype = str)
price_data_list = []

for i in range(len(ksp50)):
    try:
        price_data_list.append(pd.read_csv("./data/stock_price/"+ksp50[i]+".csv"))
        print(ksp50[i]+' load')
    except:
        print(ksp50[i]+' pass')
    if len(price_data_list)==number_of_asset:
        break

feature_deque = deque()
for j in range(2,number_of_feature+2):
    price_deque = deque()
    for i in range(number_of_asset):
        price_deque.append(price_data_list[i].to_numpy()[:,j])
    feature_deque.append(price_deque)

feature_3d_array = np.array(feature_deque,dtype=np.float32)

np.save('./preprocess/price.npy',feature_3d_array,allow_pickle=True)

v_tensor = deque()
for j in range(feature_3d_array.shape[0]):
    v_matrix = deque()
    for i in range(feature_3d_array.shape[2]-1):
        v_matrix.append(feature_3d_array[j,:,i+1]/feature_3d_array[j,:,i])
    v_tensor.append(v_matrix)
    
v_tensor = np.array(v_tensor,dtype=np.float32)