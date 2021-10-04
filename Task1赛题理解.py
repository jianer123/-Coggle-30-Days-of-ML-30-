#!/usr/bin/env python
# coding: utf-8

# # 数据读取pandas

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


## 1）载入数据集与测试集
Train_data = pd.read_csv('used_car_train_20200313.csv',sep=' ')
Test_data = pd.read_csv('used_car_testB_20200421.csv',sep=' ')

print('Train data shape:',Train_data.shape)
print('TestA data shape:',Test_data.shape)


# In[6]:


Train_data.head()


