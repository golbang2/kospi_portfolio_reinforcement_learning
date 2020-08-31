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
    data = pdr.get_data_yahoo(code+'.KS')
    return data

ksp = np.loadtxt('./data/KOSPI200.csv', delimiter=',',dtype = str)
ksp_data_lst = []

for i in ksp[:84,0]:
    time.sleep(5)
    data = price_from_yahoo(i)
    data.to_csv('./data/stock_price/'+i+'.csv')
    print(i+'done')