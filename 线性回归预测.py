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

#定义正规化函数
def normalize(x):
    mean=x.mean(axis=0)
    std=np.std(x, axis=0)
    return (x-mean)/std , mean, std

#定义cost function
def compute_cost(x, y, theta):
    m= np.size(y, axis=0)
    return np.sum(np.square(np.dot(x, theta)-y))*0.5/m


#定义梯度下降
def gradient_descent(x, y, theta, alpha, num_iters=500):
    j_plot=list()
    #求出数据数量m
    m= np.size(y, axis=0)
    #确定循环次数, 同时建立个循环次数数组备用
    iteration = range(1, num_iters)
    for iter in iteration:
        theta = theta - alpha/m*(np.dot(np.transpose(x),np.dot(x, theta)-y))
        #输出每次cost function的值以便观察是否正确迭代
        j=compute_cost(x, y,theta)
        j_plot.append(j)
        print('第'+str(iter)+'次迭代后损失函数值为: '+str(j))
    #全部循环结束之后画出j随迭代进行的变化
    plt.xlim(0, num_iters)
    plt.plot(iteration,j_plot)
    plt.xlabel('iteration')
    plt.xlabel('J')


    print('训练得到的theta结果为: ')
    print(theta)
    return theta


#正规化训练集,并提取出其中使用的mean和std,以便于预测
a1=normalize(x_tr)[0]
mean=normalize(x_tr)[1]
sigma=normalize(x_tr)[2]
theta2=np.zeros((5,1))

#在左边加上一列全为1的数列
one_1=np.ones((1,m_1)).reshape(-1,1)
a2=np.column_stack((one_1, a1))

#开始训练,并测试准确率
theta3=gradient_descent(a2, y_tr, theta2, 0.01,500)
print('训练结束后对的准确率达到: ')
accuracy(theta3, a2, y_tr)






#将训练得到的结果进行测试,首先进行正规化,然后在左边加上全为1的数列
b1= (x_te-mean)/sigma
one_1=np.ones((1,m_2)).reshape(-1,1)
b2=np.column_stack((one_1, b1))
print('对测试集进行测试的准确率为: ')
accuracy(theta3, b2, y_te)



