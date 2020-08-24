# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:29:08 2020

@author: golbang
"""


from bs4 import BeautifulSoup
import csv
import os
import re
import requests

BaseUrl='http://finance.naver.com/sise/entryJongmok.nhn?&page='

for i in range(1,22,1):
    url=BaseUrl+str(i)
    r = requests.get(url)
    soup = BeautifulSoup(r.text,'lxml')
    items=soup.find_all('td',{'class':'ctg'})
    
    for item in items:
        txt = item.a.get('href')
        k = re.search('[\d]+',txt)
        if k:
            code = k.group()
            name=item.text
            data = code,name
            with open('KOSPI200.csv','a') as f:
                writer = csv.writer(f)
                writer.writerow(data)