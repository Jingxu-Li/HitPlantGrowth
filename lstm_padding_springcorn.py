# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:42:42 2021

@author: dj079
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit 

class lstm(nn.Module):
    def __init__(self,input_size=13,hidden_size=20,output_size=1,num_layers=1):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x

model = lstm(13,50,1,2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

#数据是996*32*13，996组数据，每组32条记录，每条记录14维向量
res = np.load('./lstm_data.npy')
data = res[:,:,0:13].astype(int)
label = res[:,:,13].astype(int)
# label = label-1

# from collections import Counter
# c = Counter(label.flatten())
# print(c)

#-----------------------Data Split---------------------------------------------
split = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=41)

for train_index, test_index in split.split(np.zeros(label.shape[0]),label[:,0]):  
    data_train = data[train_index]  
    label_train = label[train_index]  
    data_test = data[test_index]  
    label_test = label[test_index]
    
#----------------------Train---------------------------------------------------
batch_size = 1
seq_len = data_train.shape[1]
input_dim = data_train.shape[2]
epoch = 10
for e in range(0,epoch):
    for i in range(0,data_train.shape[0]):
        data_batch = data_train[i].reshape(batch_size,seq_len,input_dim)
        var_x = Variable(torch.from_numpy(data_batch))
        var_y = Variable(torch.from_numpy(label_train[i]))
        out = model(var_x.float()) 
        out = torch.squeeze(out)
        loss = criterion(out.float(),var_y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss))

#---------------------Test-----------------------------------------------------
model = model.eval() # 转换成测试模式
eval_acc=0
tested_num=0
for i in range(0,data_test.shape[0]):
    data_batch = data_test[i].reshape(batch_size,seq_len,input_dim)
    var_x = Variable(torch.from_numpy(data_batch))
    # var_y = Variable(torch.from_numpy(label_test[i]))
    out = model(var_x.float()) 
    out = torch.squeeze(out)
    out = out.detach().numpy().astype(int)
    # correct = (out==label_test).sum()
    correct = np.sum(out==label_test[i])
    eval_acc += correct
    tested_num += label_test[i].shape[0]
    if(i%10==0):
        print(eval_acc/tested_num)
# print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss))

        








