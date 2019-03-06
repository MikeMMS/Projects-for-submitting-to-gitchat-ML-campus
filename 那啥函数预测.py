# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:54:15 2019

@author: nfspp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#读取文件
f=open('F:/桌面文件夹/21 天机器学习开营礼包/选修-实践项目数据/选修-实践作业/msft_stockprices_dataset.csv')
df=pd.read_csv(f)


#把最左边没用的列删掉
df2=df.drop(df.columns[0],axis=1,inplace=False)
#分割训练集和测试集,并转存为numpy数组
x_train=df2.iloc[:,[0,1,2,4]].iloc[1:700,:]
x_test=df2.iloc[:,[0,1,2,4]].iloc[700:,:]
y_train=df2.iloc[:,[3]].iloc[1:700,:]
y_test=df2.iloc[:,[3]].iloc[700:,:]
x_tr=np.array(x_train)
x_te=np.array(x_test)
y_tr=np.array(y_train)
y_te=np.array(y_test)
#求出训练集和测试集的样本数备用
m_1=np.size(x_tr,axis=0)
m_2=np.size(x_te,axis=0)


#定义正确率计算方法
def accuracy(theta, x, y):
    #设置最大允许误差
    ErrorTolerance = np.fabs(y*0.05)
    #提取总样本个数
    m= np.size(y, axis=0)
    #算出在误差范围内的预测结果个数
    num=np.sum(np.fabs(np.dot(x, theta)-y)<ErrorTolerance)
    
    acc=num/m
    print('%.4f'%acc)
    return 


#定义那啥函数(不好意思我真不知道这个中文叫啥名)
def normalEquation(x, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)),x.T), y)



#在训练集左边加上一列全为1的数列
one_1=np.ones((1,m_1)).reshape(-1,1)
x_tr=np.column_stack((one_1, x_tr))


#开始训练,并测试准确率
theta = normalEquation(x_tr, y_tr)
print('训练结束后对的准确率达到: ')
accuracy(theta, x_tr, y_tr)


#将训练得到的结果进行测试,先在左边加上全为1的数列
one_2=np.ones((1,m_2)).reshape(-1,1)
x_te=np.column_stack((one_2, x_te))
print('对测试集进行测试的准确率为: ')
accuracy(theta, x_te, y_te)







