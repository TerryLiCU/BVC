# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:15:09 2015

@author: Terry
"""

import pandas as pd
import numpy as np
import math
import scipy.stats as ss

file_location="c:/Users/Terry/Documents/Data/comex.GC_141006_151006(1min).csv"
test=pd.read_csv(file_location)

# select data for a certain period and clean out unwanted data
date=test.loc[:,'<DATE>']
df=test[date[date==20150701].index[0]:date[date==20150731].index[-1]+1]
df=df.loc[:,('<DATE>','<TIME>','<CLOSE>','<VOL>')]
df.columns=["date","time","price","volume"]
df.volume=df.volume.fillna(0)

# rearrange index
df.index=range(len(df.volume))

# check the total volume
print np.sum(df.volume)

# define function to group data accroding to volume size
def volume(co,size):
    su=0
    while su<size: 
          su=su+df.volume[co]
          co+=1
    return(su,co)
    
    
# use the function to obtain last price and volume size of each volume bar
counter=0
count=0
size=1000  # volume size
last_index=[]
volume_bar_size=[]
try:
     while count<np.nansum(df.volume)/size:
            sum=volume(counter,size)[0]
            counter=volume(counter,size)[1]
            last_index.append(counter-1)
            volume_bar_size.append(sum)
            count+=1
except:   
      pass

last_index.append(df.volume.index[-1])
volume_bar_size.append(np.nansum(df.volume[last_index[-2]+1:last_index[-1]+1]))
last_price=df.price[last_index]

# test if volume bar is correct
print np.sum(df.volume)
print np.sum(volume_bar_size)

price_volumebar=np.zeros((len(last_index),2))
price_volumebar=pd.DataFrame(price_volumebar,index=last_index,columns=['price','bar_size'])
price_volumebar.price=df.price[last_index]
price_volumebar.bar_size=volume_bar_size

# volume-weighted standard deviation
def weighted_std(values, weights):
   mean = np.nanmean(values)
   weight_variance = weights*((values-mean)**2)
   weight_sum=np.nansum(weights)
   variance_sum=np.nansum(weight_variance)
   stDev=math.sqrt(variance_sum/weight_sum)
   return (stDev)

  
d_price_diff=np.diff(price_volumebar.price)
d_price_diff_percentage=d_price_diff/price_volumebar.price[0:-1]
stDev=weighted_std(d_price_diff,price_volumebar.bar_size[1:])

buy_sell=np.zeros((len(d_price_diff),4))
buy_sell=pd.DataFrame(buy_sell,columns=['buy','sell','total','label'])

# Applying BVC algorithm
for i in range(0,len(d_price_diff)): 
    buy_sell.loc[i,'buy']=price_volumebar.bar_size[last_index[i+1]]*ss.t.cdf(d_price_diff[i]/stDev,0.05)
    buy_sell.loc[i,'sell']=price_volumebar.bar_size[last_index[i+1]]-buy_sell.buy[i]
    buy_sell.loc[i,'total']=price_volumebar.bar_size[last_index[i+1]]

# If more buys, then label the var as 'buy'
for j in range(0,len(d_price_diff)):
    if buy_sell.buy[j]-buy_sell.sell[j]>0:
        buy_sell.loc[j,'label']='buy'
    else:
        buy_sell.loc[j,'label']='sell'


# caculate VPIN
buy_sell_vspread=buy_sell.buy-buy_sell.sell
#vpin=np.nansum(abs(buy_sell.buy-buy_sell.sell))/np.nansum(buy_sell.total)
rolling_size=1 # rolling wondow size in terms of how many volume bars
vpin=pd.rolling_sum(abs(buy_sell_vspread),rolling_size)/pd.rolling_sum(buy_sell.total,rolling_size)
rstd=pd.rolling_std(last_price,rolling_size)
# plot of VPIN vs Price
import matplotlib.pyplot as plt

# corresponded time index
time_bar_date=df.date[last_index]
b1=[str(i) for i in time_bar_date]
time_bar_time=df.time[last_index]
b2=[str(i) for i in time_bar_time]

b=['' for x in range(len(b1))]
for i in range(0,len(b1)):
     b[i]=''.join([b1[i],'-',b2[i]])
     
a=range(len(vpin))
a=[float(i) for i in a]

fig,ax1=plt.subplots()
ax1.plot(vpin,color='red',label='VPIN')
ax1.set_ylabel('VPIN',color='red',fontsize=18)
ax1.set_xticks(a,minor=False)
ax1.set_xticklabels(b, rotation=60)
ax1.yaxis.grid()
ax1.xaxis.grid()
ax2=ax1.twinx()
ax2.plot(last_price,color='blue',label='sell volume')
ax2.set_ylabel('Price',color='blue',fontsize=18)


fig,ax1=plt.subplots()
ax1.plot(d_price_diff_percentage,color='red',label='VPIN')
ax1.set_ylabel('Price difference',color='red',fontsize=18)
ax1.set_xticks(a,minor=False)
ax1.set_xticklabels(b, rotation=60)
ax2=ax1.twinx()
ax2.plot(last_price,color='blue',label='sell volume')
ax2.set_ylabel('Price',color='blue',fontsize=18)

## Look at the volume spread vs price movement
#coefficients = np.polyfit(d_price_diff, buy_sell_vspread,1)
#polynomial = np.poly1d(coefficients)
#y=buy_sell_vspread
#y=y[y!=min(y)]
#x=d_price_diff[y.index]
#xs=sorted(x)
#ys=sorted(polynomial(x))

#
#fig=plt.figure()
#plt.plot(buy_sell_vspread,d_price_diff,'o')
#slope, intercept, r_value, p_value, std_err = ss.linregress(d_price_diff, buy_sell_vspread)
#print "r-squared:", r_value**2
##ax2=ax1.twinx()
##ax2.plot(xs,ys,color='blue')
#
## calculate the cdf vpin
#vpin=vpin[0:1148]
#sorted_vpin=np.sort(vpin)
#yvals=1.*np.arange(len(sorted_vpin))/(len(sorted_vpin)-1)
#index=np.argsort(vpin)
#cdf_vpin=vpin
#cdf_vpin[index]=yvals
#
#fig,ax1=plt.subplots()
#ax1.plot(cdf_vpin,color='red',label='VPIN')
#ax1.set_ylabel('Price difference',color='red',fontsize=18)
#ax1.set_xticks(a,minor=False)
#ax1.set_xticklabels(b, rotation=60)
#ax2=ax1.twinx()
#ax2.plot(last_price,color='blue',label='sell volume')
#ax2.set_ylabel('Price',color='blue',fontsize=18)

# To calculate the actual buy and sell volume
#actual_buy_sell=np.zeros((len(last_index),3))
#actual_buy_sell=pd.DataFrame(actual_buy_sell,columns=['buy','sell','total'])
#
#for i in range(0,len(last_index)):
#    if i==0:
#       actual_buy_sell.buy[i]=np.nansum(df.volume[0:last_index[i]][df.bid_ask[0:last_index[i]]=='BID'])
#       actual_buy_sell.sell[i]=np.nansum(df.volume[0:last_index[i]][df.bid_ask[0:last_index[i]]=='ASK'])
#       actual_buy_sell.total[i]=np.nansum(df.volume[0:last_index[i]])
#    if i>0:
#        actual_buy_sell.buy[i]=np.nansum(df.volume[last_index[i-1]+1:last_index[i]][df.bid_ask[last_index[i-1]+1:last_index[i]]=='BID'])
#        actual_buy_sell.sell[i]=np.nansum(df.volume[last_index[i-1]+1:last_index[i]][df.bid_ask[last_index[i-1]+1:last_index[i]]=='ASK'])
#        actual_buy_sell.total[i]=np.nansum(df.volume[last_index[i-1]+1:last_index[i]])
#
#
## test the accuracy of BVC
#bulk=np.zeros((len(d_price_diff),1))
#for i in range(0,len(d_price_diff)):
#    bulk[i]=min(buy_sell.buy[i],actual_buy_sell.buy[i+1])+min(buy_sell.sell[i],actual_buy_sell.sell[i+1])
#
#a_ratio=np.nansum(bulk)/np.nansum(buy_sell.total)





#ax2.set_xticklabels(time_bar_index)

#plt.figure(1)
#plt.plot(vpin,color='red',label='VPIN')
#legend=plt.legend(loc="upper left")
#plt.plot(last_price/10000,color='blue',label='sell volume')
#legend=plt.legend(loc='upeer left')
#
#plt.figure(2)
#plt.plot(buy_sell_vspread,color='green',label='volume spread')
