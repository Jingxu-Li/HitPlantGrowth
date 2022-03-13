# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:58:27 2021

@author: dj079
"""

import numpy as np
# import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
# from pylab import *
from collections import Counter
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
from xgboost import plot_importance
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


#3-旬数 5-生长期名称 7-发育程度 8-发育期距平 12-日平均气温 13-积温距平 15-10cm 16-20cm 17-50cm
#10 - 生长状况

sc = np.load('./winter_wheat.npy')

# # 旬数处理
s1 = np.int32(sc[:,2])
s2 = np.int32(sc[:,3])
s2 = s1.dot(3) + s2
sc[:,3] = s2

# 生长期名称处理
# a = np.where(sc[:,5]=='-9999')
# sc[a] = '0'
sc[np.where(sc[:,5]=='未进'),5] = 0
sc[np.where(sc[:,5]=='播种'),5] = 1
sc[np.where(sc[:,5]=='出苗'),5] = 2
sc[np.where(sc[:,5]=='分蘖'),5] = 3
sc[np.where(sc[:,5]=='停止生长'),5] = 4
sc[np.where(sc[:,5]=='返青'),5] = 5
sc[np.where(sc[:,5]=='拔节'),5] = 6
sc[np.where(sc[:,5]=='抽穗'),5] = 7
sc[np.where(sc[:,5]=='乳熟'),5] = 8
sc[np.where(sc[:,5]=='成熟'),5] = 9

# 发育程度处理
sc[np.where((sc[:,7]=='-9999')|(sc[:,7]=='普遍期')),7] = 2
sc[np.where(sc[:,7]=='开始期'),7] = 1
sc[np.where(sc[:,7]=='末期'),7] = 3

# 发育期距平处理
for recs in sc:
    if('提前' in recs[8]):
        recs[8] = -1
    elif(('99' in recs[8]) or ('正常' in recs[8])):
        recs[8] = 0
    elif('推迟' in recs[8]):
        recs[8] = 1
        
# 日平均气温处理
# sc[np.where(sc[:,12]=='-9999')] = '0'
# sc[np.where(sc[:,13]=='-9999')] = '0'
# sc[np.where(sc[:,15]=='-9999')] = '0'
# sc[np.where(sc[:,16]=='-9999')] = '0'
# sc[np.where(sc[:,17]=='-9999')] = '0'
# sc[np.where(sc[:,18]=='-9999')] = '0'
# sc[np.where(sc[:,19]=='-9999')] = '0'
# #生长状况处理
# sc[np.where(sc[:,10]=='-9999')] = '0'
# sc[np.where(sc[:,10]=='-999')] = '0'
# sc[np.where(sc[:,10]=='4')] = '0'
# sc[np.where(sc[:,10]=='3')] = '0'

res = sc[np.where(sc[:,0]!='0')]
np.save('./winter_wheat_res.npy',res)

res = np.load('./winter_wheat_res.npy')

data = res[:,[0,1,3,5,7,9,11,12,13,14,15,16,17]].astype(int)
label = res[:,8].astype(int)
label = label+1

#-----------------------Data Split---------------------------------------------
split = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=41)

for train_index, test_index in split.split(np.zeros(label.shape[0]), label):  
    data_train = data[train_index]  
    label_train = label[train_index]  
    data_test = data[test_index]  
    label_test = label[test_index]
#-----------------------SVM----------------------------------------------------
# svc = SVC(kernel='rbf', C=1000)
# svc.fit(data_train, label_train)
# score = svc.score(data_test, label_test)
# print("accuracy of svm: %.2f%%"%(score*100))
# clf = tree.DecisionTreeClassifier(criterion='gini',splitter='best')
# clf.fit(data_train,label_train)
# ans = clf.score(data_test,label_test)
# print("accuracy of DT: %.2f%%"%(ans*100.0))

# clf = GradientBoostingClassifier(n_estimators=500, learning_rate = 0.3)
# clf.fit(data_train,label_train)
# ans = clf.score(data_test,label_test)


# ans = clf.predict(data_test)
# print("accuracy of GBDT: %.2f%%"%(np.mean(ans==label_test)*100.0))

# ans = clf.score(data_test,label_test)
# xgb_params = {'max_depth':[4,5,6,7],'learning_rate':np.linspace(0.03,0.3,10),'n_estimators':[100,150,200]}
# model = XGBClassifier(max_depth = 6,learning_rate = 0.06,n_estimators = 150,use_label_encoder = False)

#------------------------------------------------------------------------------
model = XGBClassifier(n_estimators = 1000,learning_rate = 0.02,use_label_encoder = False)
model.fit(data_train,label_train,eval_metric = 'auc')
label_pred = model.predict(data_test)
# xgb_search.fit(data_train,label_train,eval_metric = 'auc')

# print(xgb_search.grid_scores_)
# print(xgb_search.best_params_)
# print(xgb_search.best_score_)
accuracy = accuracy_score(label_test, label_pred)
print("accuracy of xgboost: %.2f%%" %(accuracy*100.0))

# plot_tree(model, num_trees = 0)
plot_importance(model)
plt.show()



















