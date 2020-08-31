# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:28:00 2020

@author: golbang
"""

number_of_asset = 10
number_of_feature = 4 # close,high,low,volume
train_set_length = 800
test_set_length = 200

import numpy as np
import pandas as pd
from collections import deque

#load kospi200 list
read_csv_file = 'KOSPI200.csv'
ksp_list = np.loadtxt('./data/'+read_csv_file, delimiter=',',dtype = str)
data_deque = deque()

for i in ksp_list:
    ksp_data = pd.read_csv("./data/stock_price/"+i[0]+".csv")
    if len(ksp_data)>train_set_length+test_set_length:
        ksp_data = ksp_data[['Close','High','Low','Volume']].to_numpy(dtype=np.float32)
        data_deque.append(ksp_data)
        print(i[1]+' loaded, length : '+str(len(ksp_data)))
    if len(data_deque) == number_of_asset:
        break
    
data_tensor = np.array(data_deque,dtype=np.float32)
np.save('./preprocess/train_price_tensor.npy',data_tensor[:,-train_set_length-test_set_length:-test_set_length,:],allow_pickle=True)
np.save('./preprocess/test_price_tensor.npy',data_tensor[:,-test_set_length:,:],allow_pickle=True)
