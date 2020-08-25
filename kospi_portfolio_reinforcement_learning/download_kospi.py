import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
import bs4
import datetime as dt
import pandas as pd
from collections import deque
import random
import time


def date_format(date):
    date=str(date).replace('-','.')
    yyyy=int(date.split('.')[0])
    mm=int(date.split('.')[1])
    dd=int(date.split('.')[2])
    date=dt.date(yyyy,mm,dd)
    return date

def extract_data(pages,index):
    for j in range(1,pages+1):
        time.sleep(0.2)
        page=j
        naver='https://finance.naver.com/item/sise_day.nhn?code='+index+'&page='+str(page)
        source=urlopen(naver).read()
        source=bs4.BeautifulSoup(source,'lxml')
        dates=source.find_all('span',class_='tah p10 gray03')
        closes=source.find_all('span',class_='tah p11')
        for i in range(len(closes)-1,-1,-1):
            if str(closes[i])=='<span class="tah p11">0</span>':
                del closes[i]
        if j ==1:
            date=[]
            close=[]
            volume=[]
            Open=[]
            high=[]
            low=[]
        for i in range(len(dates)):
            date.append(dates[i].text)
            date[(j-1)*10+i]=date_format(date[(j-1)*10+i])
            close.append(closes[(i)*5].text)
            close[(j-1)*10+i]=close[(j-1)*10+i].replace(',','')
            close[(j-1)*10+i]=float(close[(j-1)*10+i])
            volume.append(closes[(i)*5+4].text)
            volume[(j-1)*10+i]=volume[(j-1)*10+i].replace(',','')
            volume[(j-1)*10+i]=float(volume[(j-1)*10+i])
            Open.append(closes[(i)*5+1].text)
            Open[(j-1)*10+i]=Open[(j-1)*10+i].replace(',','')
            Open[(j-1)*10+i]=float(Open[(j-1)*10+i])
            high.append(closes[(i)*5+2].text)
            high[(j-1)*10+i]=high[(j-1)*10+i].replace(',','')
            high[(j-1)*10+i]=float(high[(j-1)*10+i])
            low.append(closes[(i)*5+3].text)
            low[(j-1)*10+i]=low[(j-1)*10+i].replace(',','')
            low[(j-1)*10+i]=float(low[(j-1)*10+i])
    dataframe={'Open' : pd.Series(Open, index=date),'close' : pd.Series(close, index=date),'high' : pd.Series(high, index=date),'low' : pd.Series(low, index=date),'volume' : pd.Series(volume, index=date)}
    dataframe=pd.DataFrame(dataframe)
                
    return dataframe

ksp = np.loadtxt('./data/KOSPI200.csv', delimiter=',',dtype = str)
ksp_data_lst = []

for i in ksp[51:]:
    try:
        ksp_data_lst.append(extract_data(80,i))
        ksp_data_lst[-1].to_csv('d:/kospi_data_80pg/'+i+'.csv')
        print(i+'done')
    except:
        print(i+'failed')