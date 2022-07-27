from pickle import FALSE
from tkinter import Label
from tkinter.ttk import LabeledScale
import tensorcircuit as tc
import tensorflow as tf
import math
import numpy as np
import itertools
from matplotlib import pyplot as plt
import tensorcircuit as tc
import random
K = tc.set_backend("tensorflow")
X = tc.gates._x_matrix  # same as tc.gates.xgate().tensor.numpy()
Y = tc.gates._y_matrix  # same as tc.gates.ygate().tensor.numpy()
Z = tc.gates._z_matrix  # same as tc.gates.zgate().tensor.numpy()
H = tc.gates._h_matrix  # same as tc.gates.hgate().tensor.numpy()
S = tc.gates._s_matrix
T = tc.gates._t_matrix

#初始化行数、列数
L=3
R=3


#随机初始化E
E=np.random.randint(2,size=R*(L-1)+L*(R-1)) 

#由E初始化竖向边
LL = np.zeros([0],dtype=int)
for i in range(R*(L-1)):
        if E[i] == 1:
            LL = np.append(LL,[i])
LL = np.append(LL,[R*L])
LL = np.append(LL,[R*L+1])
LL_new = np.delete(LL,[-1])
LL_s = LL.size
LLL=LL.tolist()
LLL_new=LL_new.tolist()
#LLL=range(LL_s)
#LLL_new=range(LL_s-1)
#for i in range(LL_s):
#    LLL[i]=LL[i]
#for i in range(LL_s-1):
#    LLL_new[i]=LL_new[i]
#print(LL)
#print(LL_new) 

#由E初始化横向边
RR=np.zeros([0],dtype=int)
for i in range(L):
    for j in range(R-1):
        if E[R*(L-1)+i*(R-1)+j] == 1:
            RR = np.append(RR,[i*R+j])
RR = np.append(RR,[R*L])
RR_new = np.delete(RR,[-1])
RR_s = RR.size
RRR=RR.tolist()
RRR_new=RR_new.tolist()
#RRR=range(RR_s)
#RRR_new=range(RR_s-1)
#for i in range(LL_s):
#    RRR[i]=RR[i]
#for i in range(RR_s-1):
#    RRR_new[i]=RR_new[i]
#print(RR)
#print(RR_new)

#可视化
def pri_1(i):
    print(int(r[i,0]),end=' ')
    for j in range(R-1):
        if E[R*(L-1)+i*(R-1)+j] == 1:
            print("-",end=' ')
        else:
            print(" ",end=' ')
        print(int(r[i,j+1]),end=' ')
    print('')
def pri_2(i):
    for j in range(R):
        if E[i*R+j] == 1:
            print('|'," ",end=' ')
        else:
            print(" "," ",end=' ')
    print('')
def pri_3(i):
    print("?",end=' ')
    for j in range(R-1):
        if E[R*(L-1)+i*(R-1)+j] == 1:
            print("-",end=' ')
        else:
            print(" ",end=' ')
        print("?",end=' ')
    print('')


#解个数不明的Grover搜索算法
def mainloop():
    #检查边集是否为空
    if (LL_s+RR_s-3 == 0):
        return "All state"
    
    else:
        #判断函数，在第1辅助比特为±（0-1）的情况下可以输出c关于α（所有非解的均匀叠加态）翻转
        def oracle(c):
            #比较横向边，将i*R+j+1与i*R+j的比较结果存储在i*R+j上
            for i in range(L):
                for j in range(R-1):
                    if E[(L-1)*R+i*(R-1)+j] == 1:
                        c.CNOT(i*R+j+1,i*R+j)
            
            #将横向边比较的总结果存储在第0辅助比特上（将X门作用于第0辅助比特）
            if RR_s==1:
                c.X(RR[-1])
            else:
                c.multicontrol(*RRR,ctrl=[1 for __ in RR_new],unitary=X)
            
            #逆向恢复
            for i in range(L):
                for j in range(R-1):
                    if E[(L-1)*R+(L-1-i)*(R-1)+(R-2-j)] == 1:
                        c.CNOT((L-1-i)*R+(R-2-j)+1,(L-1-i)*R+(R-2-j))
            
            #比较纵向边，将i+R与i与的比较结果存储在i上
            for i in range((L-1)*R):
                if E[i] == 1:
                    c.CNOT(i+R,i)
            
            #将纵向边比较和第0辅助比特的总结果作用在第1辅助比特上（将X门作用于第1辅助比特）
            c.multicontrol(*LLL,ctrl=[1 for __ in LL_new],unitary=X)
            
            #逆向恢复
            for i in range((L-1)*R):
                if E[(L-1)*R-1-i] == 1:
                    c.CNOT(L*R-1-i,(L-1)*R-1-i)
            
            #逆向恢复
            for i in range(L):
                for j in range(R-1):
                    if E[(L-1)*R+i*(R-1)+j] == 1:
                        c.CNOT(i*R+j+1,i*R+j)
            
            #逆向恢复
            if RR_s==1:
                c.X(RR[-1])
            else:
                c.multicontrol(*RRR,ctrl=[1 for __ in RR_new],unitary=X)
            
            #逆向恢复
            for i in range(L):
                for j in range(R-1):
                    if E[(L-1)*R+(L-1-i)*(R-1)+(R-2-j)] == 1:
                        c.CNOT((L-1-i)*R+(R-2-j)+1,(L-1-i)*R+(R-2-j))
            
            return c

        #输出c关于φ（解与非解的均匀叠加态）的翻转
        def reflect(c):
            for i in range(R*L):
                c.H(i)
                c.X(i)
    
            c.multicontrol(*range(R*L),ctrl=[1 for _ in range(R*L-1)],unitary=Z)
    
            for i in range(R*L):
                c.X(i)
                c.H(i)
    
            return c

        #输出一次转动的结果，同时包含第1辅助比特的恢复
        def Grover(c):
            c = c
            c.X(R*L+1)
            c.H(R*L+1)
            c = oracle(c)
            c = reflect(c)
            c.H(R*L+1)
            c.X(R*L+1)
    
            return c

        #制备初始输入叠加态
        def initialize():
            c=tc.Circuit(R*L+2)
            for i in range(R*L):
                c.H(i)
            return c

        #利用oracle判断纯态c是否为解
        def classy(c):
            c = oracle(c)
            cm = c.measure(R*L+1, with_prob=False)
            if cm[0].numpy().sum() == 1:
                return 1
            else:
                return 0
        
        #初始化主循环与可视化所需的参数
        m=1
        m_=1.1
        suc=False
        round = 0

        while suc == False:
            #在[0,m]中随机取非负整数j，作为grover函数的执行次数
            j=random.randint(0,int(m))
            print("Round:",round,"(m,j)=","(",m,j,")")

            #执行grover函数并且测量
            cmm=np.arange(R*L+1)
            c=initialize()
            for i in range(j):
                c=Grover(c)
            r=c.measure(*range(R*L+2),with_prob=False)
            
            #将测量结果制为纯态
            cmm=K.numpy(r[0])
            sum=0
            cmminput=np.zeros([2**(R*L+2)])
            for i in range(R*L+2):
                sum=sum+(cmm[i]*2**(R*L+1-i))
            cmminput[int(sum)]=1
            cmmm=tc.Circuit(R*L+2,inputs=cmminput)
            
            #判断测量结果
            if classy(cmmm) == 1:
                suc == True
                np.delete(cmm,[-1,-2])
                return np.delete(cmm,[-1,-2])
            
            #放大m以提高测量结果正确率、维持理论次数期望的同时，给予grover函数执行次数上限以减少实际执行次数
            m = min(m*m_,math.sqrt(2**(R*L)))
            round = round + 1

#可视化初始边
for i in range(L-1):
    pri_3(i)
    pri_2(i)
pri_3(L-1)

#可视化结果
r=mainloop()
r=r.reshape(L,R)
for i in range(L-1):
    pri_1(i)
    pri_2(i)
pri_1(L-1)