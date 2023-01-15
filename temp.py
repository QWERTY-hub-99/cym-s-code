# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf    #偏自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox    #白噪声检验
from statsmodels.tsa.arima.model import ARIMA
import itertools
import datetime

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


stock_code = ['600519','300896','300751','300750','605117']
'''000300 :  沪深300
   600519 :  贵州茅台
   300896 ： 爱美客
   300751 ： 迈为股份
   300750 ： 宁德时代
   605117 ： 德业股份
   603290 ： 斯达半导
   300760 ： 迈瑞医疗
   000596 ： 古井贡酒
'''
stock_code1 =  ['300896','300751','300750','605117']
path = 'E://研一第一学期作业//时间序列期末报告//时间序列//'

for i in stock_code:
    locals()['price_'+i] = pd.read_excel(path + i + '.xlsx')
    l = len(locals()['price_'+i])
    for j in range(l):
        locals()['price_'+i].iloc[j,0] = pd.to_datetime(locals()['price_'+i].iloc[j,0][0:10])
    
    locals()['price_'+i].index = locals()['price_'+i].iloc[:,0]
    locals()['price_'+i] = locals()['price_'+i].drop(columns=['时间'])

    locals()['price_'+i]  = locals()['price_'+i].loc['2022']
    locals()['price_'+i] = locals()['price_'+i].drop(columns=['涨幅'])
    locals()['price_'+i] = np.array( locals()['price_'+i])
    
    
    
price_000300 = pd.read_excel(path + '000300' + '.xlsx')
l = len(price_000300)
for j in range(l):
    price_000300.iloc[j,0] = pd.to_datetime(price_000300.iloc[j,0][0:10])

price_000300.index = price_000300.iloc[:,0]
price_000300 = price_000300.drop(columns=['时间'])

price_000300 = price_000300.loc['2022']
price_000300 = price_000300.drop(columns=['涨幅'])
data = price_600519
for i in stock_code1:
    data = np.concatenate((data,locals()['price_'+i]),axis = 1)
    
    
l = len(price_600519)
rho = 0.618
m = 5
alpha = 0.999/(rho*m)
epsilon = 1e-4
max_iteration = int(1e4)
gamma = 0.01 
lambd = 10*gamma

profit = np.zeros(l)


one =  np.ones(shape = [m,1])

bt = one/m

for i in range(l):
    x=np.array([data[i,:]]).T
    f1 = 1./x
    bk = one/m
    ksi = 10
    
    for k in range(max_iteration):
        bkk = bk
        B = bkk - alpha*one*ksi - alpha*rho*one*(bkk.sum()- 1) + alpha*f1 - bt
        temp = abs(B)-alpha*lambd
        for k in range(len(temp)):
            if temp[k] < 0:
                temp[k] = 0
        bk = bt + np.sign(B)*temp
        for k in range(len(bk)):
            if bk[k] < 0:
                bk[k] = 0
        ksi = ksi + rho*(bk.sum()-1)
        if np.linalg.norm(bk - bkk)/np.linalg.norm(bk) <= epsilon:
            break
    bt = bk
    port = bk/bk.sum()
    print(port)
    profit[i] = (port * x).sum()
    
profit = pd.DataFrame(profit)
profit = profit.pct_change(1).dropna()
price_000300 = price_000300.pct_change(1).dropna()
#price_000300.drop(price_000300.head(1).index,inplace=True)
profit.index = price_000300.index
plt.plot(profit,label = 'TCR')
plt.plot(price_000300,label = 'CSI 300 ')
plt.legend()
plt.title('TCR model v.s. CSI 300')
plt.savefig(path+'aaaaaa.jpg',dpi = 300)

l1 =1
l2 =1
for i in range(241):
    l1 = l1 *(1+ profit.iloc[i].values)
   
for i in range(241):
    l2 = l2 *(1+ price_000300.iloc[i].values)
    
print(l1/l2)