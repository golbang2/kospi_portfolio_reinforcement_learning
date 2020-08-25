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

read_csv_file = 'KOSPI200.csv'

ksp_list = np.loadtxt('./data/'+read_csv_file, delimiter=',',dtype = str)
price_data_list = []

for i in range(len(ksp_list)):
    try:
        price_data_list.append(pd.read_csv("./data/stock_price/"+ksp_list[i]+".csv"))
        print(ksp_list[i]+' load')
    except:
        print(ksp_list[i]+' pass')

feature_deque = deque()
for j in range(2,number_of_feature+2):
    price_deque = deque()
    for i in range(number_of_asset):
        price_deque.append(price_data_list[i].to_numpy()[:,j])
    feature_deque.append(price_deque)

feature_3d_array = np.array(feature_deque,dtype=np.float32)

np.save('./preprocess/price_tensor.npy',feature_3d_array,allow_pickle=True)