# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:37:15 2021

@author: golbang
"""


import numpy as np
from urllib.request import urlopen
import bs4
import datetime as dt
import pandas as pd
from collections import deque
import time
import pandas_datareader as pdr

def date_format(date):
    date=str(date).replace('-','.')
    yyyy=int(date.split('.')[0])
    mm=int(date.split('.')[1])
    dd=int(date.split('.')[2])
    date=dt.date(yyyy,mm,dd)
    return date

def price_from_yahoo(code):
    data = pdr.get_data_yahoo(code+'.KS',start = date_format('2020-09-1'),end = date_format('2020-12-20'))
    return data

ksp = np.loadtxt('d:/data/KOSPI200an.csv', delimiter=',',dtype = str)
ksp_data_lst = []

for i in ksp:
    new_data = price_from_yahoo(i[0])
    #data.to_csv('d:/data/stock_price/'+i+'.csv')
    old_data = pd.read_csv('./data/stock_price/'+i[0]+'.csv',index_col = 'Date')
    concat_data = pd.concat([old_data,new_data])
    concat_data.to_csv('./data/stock_price_extended/'+i[0]+'.csv')
    time.sleep(3)