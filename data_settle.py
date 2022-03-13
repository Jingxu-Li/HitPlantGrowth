# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 21:57:40 2021

@author: dj079
"""

#-------------------------Data loading-----------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
count = 0
data_num = 0
data_sp = []
path = os.listdir('.//dataset')
for prov in path:
    txtToRead = os.listdir(".//dataset//" + prov)
    f = open(".//dataset//" + prov + "//" + txtToRead[0],'r',encoding='utf-8')
    line = f.readline() # read title
    while(line):
        line = f.readline()
        line_split = line.split(" ")
        if(line_split!=['']):
            data_sp.append(line_split)
            data_num = data_num + 1
    f.close()
print("datanum:%d"%(data_num))
#-------------------------Data settling----------------------------------------   
 
spring_wheat = []
soybeans = []
background = []
spring_corn = []
summer_corn = []
rice1 = []
cane = []
rice2 = []
rice3 = []
winter_wheat = []
sorghum = []
grass = []
ratoon_cane = []
potato = []
rape = []
cotton = []
peanut = []
sweet_potato = []
peanut = []
glutinous_rice = []
other = []
ordinary_cotton = []
tobacco = []
set_corn = []
barley = []
beet = []

for recs in range(len(data_sp)):
    if data_sp[recs][4] == '春小麦':
        spring_wheat.append(data_sp[recs])
    elif data_sp[recs][4] == '大麦':
        barley.append(data_sp[recs])
    elif data_sp[recs][4] == '冬小麦':
        winter_wheat.append(data_sp[recs])  
    elif data_sp[recs][4] == '大豆':
        soybeans.append(data_sp[recs])
    elif data_sp[recs][4] == '春玉米':
        spring_corn.append(data_sp[recs])
    elif data_sp[recs][4] == '套玉米':
        set_corn.append(data_sp[recs])
    elif data_sp[recs][4] == '夏玉米':
        summer_corn.append(data_sp[recs])
    elif data_sp[recs][4] == '白地':
        background.append(data_sp[recs])
    elif data_sp[recs][4] == '一季稻':
        rice1.append(data_sp[recs])
    elif data_sp[recs][4] == '甘蔗':
        cane.append(data_sp[recs])     
    elif data_sp[recs][4] == '晚稻':
        rice2.append(data_sp[recs])   
    elif data_sp[recs][4] == '早稻':
        rice3.append(data_sp[recs])
    elif data_sp[recs][4] == '高粱':
        sorghum.append(data_sp[recs])
    elif data_sp[recs][4] == '牧草':
        grass.append(data_sp[recs])
    elif data_sp[recs][4] == '宿根蔗':
        ratoon_cane.append(data_sp[recs])
    elif data_sp[recs][4] == '马铃薯':
        potato.append(data_sp[recs])
    elif data_sp[recs][4] == '甘薯':
        sweet_potato.append(data_sp[recs])
    elif data_sp[recs][4] == '花生':
        peanut.append(data_sp[recs])
    elif data_sp[recs][4] == '棉花':
        cotton.append(data_sp[recs])
    elif data_sp[recs][4] == '普通棉':
        ordinary_cotton.append(data_sp[recs])
    elif data_sp[recs][4] == '糯稻':
        glutinous_rice.append(data_sp[recs])
    elif data_sp[recs][4] == '油菜':
        rape.append(data_sp[recs])
    elif data_sp[recs][4] == '烟草':
        tobacco.append(data_sp[recs])
    elif data_sp[recs][4] == '甜菜':
        beet.append(data_sp[recs])
    elif data_sp[recs][4] == '-999.0' or data_sp[recs][4] == '-9999' or data_sp[recs][4] == '其它作物':
        continue
    else:
        count = count + 1
        other.append(data_sp[recs])
    
#np.save('./spring_corn/spring_corn.npy',spring_corn)
#np.save('./winter_wheat/winter_wheat.npy',winter_wheat)
np.save('rape.npy',rape)#油菜
np.save('rice1.npy',rice1)#一季稻
np.save('rice2.npy',rice2)#晚稻
np.save('rice3.npy',rice3)#早稻
np.save('spring_wheat.npy',spring_wheat)#春小麦
np.save('summer_corn.npy',summer_corn)#夏玉米
np.save('winter_wheat.npy',winter_wheat)#冬小麦

# Collect number of examples
#%%
from collections import Counter
spring_corn_count = np.array(spring_corn)
label = spring_corn_count[:,5]
c = Counter(label)
print("-----------春玉米------------")
print(c)

rape_count = np.array(rape)
label = rape_count[:,5]
c = Counter(label)
print("-----------油菜------------")
print(c)

rice1_count = np.array(rice1)
label = rice1_count[:,5]
c = Counter(label)
print("-----------一季稻------------")
print(c)

rice2_count = np.array(rice2)
label = rice2_count[:,5]
c = Counter(label)
print("-----------晚稻------------")
print(c)

rice3_count = np.array(rice3)
label = rice3_count[:,5]
c = Counter(label)
print("-----------早稻------------")
print(c)

spring_wheat_count = np.array(spring_wheat)
label = spring_wheat_count[:,5]
c = Counter(label)
print("-----------春小麦------------")
print(c)

summer_corn_count = np.array(summer_corn)
label = summer_corn_count[:,5]
c = Counter(label)
print("-----------夏玉米------------")
print(c)

winter_wheat_count = np.array(winter_wheat)
label = winter_wheat_count[:,5]
c = Counter(label)
print("-----------冬小麦------------")
print(c)