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

L=2
R=2

E=np.random.randint(2, size=R*(L-1)+L*(R-1))

LL = np.zeros([0],dtype=int)
for i in range(R*(L-1)):
        if E[i] == 1:
            LL = np.append(LL,[i])
LL = np.append(LL,[R*L])
LL = np.append(LL,[R*L+1])
LL_new = np.delete(LL,[-1])
LL_s = LL.size
LLL=LL.flat
LLL_new=LL_new.flat
#LLL=range(LL_s)
#LLL_new=range(LL_s-1)
#for i in range(LL_s):
#    LLL[i]=LL[i]
#for i in range(LL_s-1):
#    LLL_new[i]=LL_new[i]
print(LL)
print(LL_new)

RR=np.zeros([0],dtype=int)
for i in range(L):
    for j in range(R-1):
        if E[R*(L-1)+i*(R-1)+j] == 1:
            RR = np.append(RR,[i*R+j])
RR = np.append(RR,[R*L])
RR_new = np.delete(RR,[-1])
RR_s = RR.size
RRR=RR.flat
RRR_new=RR_new.flat
#RRR=range(RR_s)
#RRR_new=range(RR_s-1)
#for i in range(LL_s):
#    RRR[i]=RR[i]
#for i in range(RR_s-1):
#    RRR_new[i]=RR_new[i]
print(RR)
print(RR_new)

def oracle(c):
    for i in range(L):
        for j in range(R-1):
            if E[i*R+j] == 1:
                c.CNOT(i*R+j+1,i*R+j)
    
    if RR_s==1:
        c.X(RR[-1])
    else:
        c.multicontrol(*RRR,ctrl=[1 for __ in RR_new],unitary=X)
    
    for i in range(L):
        for j in range(R-1):
            if E[(L-1-i)*R+(R-2-j)] == 1:
                c.CNOT((L-1-i)*R+(R-2-j)+1,(L-1-i)*R+(R-2-j))
    
    for i in range((L-1)*R):
        if E[i] == 1:
            c.CNOT(i+R,i)
    #print(*LLL)
    c.multicontrol(*LLL,ctrl=[1 for __ in LL_new],unitary=X)
    
    for i in range((L-1)*R):
        if E[(L-1)*R-1-i] == 1:
            c.CNOT(L*R-1-i,(L-1)*R-1-i)
    
    for i in range(L):
        for j in range(R-1):
            if E[i*R+j] == 1:
                c.CNOT(i*R+j+1,i*R+j)
    
    if RR_s==1:
        c.X(RR[-1])
    else:
        c.multicontrol(*RRR,ctrl=[1 for __ in RR_new],unitary=X)
    
    for i in range(L):
        for j in range(R-1):
            if E[(L-1-i)*R+(R-2-j)] == 1:
                c.CNOT((L-1-i)*R+(R-2-j)+1,(L-1-i)*R+(R-2-j))
    
    return c

def reflect(c):
    for i in range(R*L):
        c.H(i)
        c.X(i)
    
    c.multicontrol(*range(R*L),ctrl=[1 for _ in range(R*L-1)],unitary=Z)
    
    for i in range(R*L):
        c.X(i)
        c.H(i)
    
    return c


def Groove(c):
    c = c
    c.X(-1)
    c.H(-1)
    c = oracle(c)
    c = reflect(c)
    c.H(-1)
    c.X(-1)
    
    return c

def initialize():
    c=tc.Circuit(R*L+2)
    for i in range(R*L):
        c.H(i)
    return c

def classy(c):
    c = oracle(c)
    cm = c.measure(R*L, with_prob=False)
    if cm[0].numpy().sum() == 1:
        return 1
    else:
        return 0

def mainloop():
    if (LL_s+RR_s-3 == E.size):
        return "All state"
    
    else:
        m=1
        m_=1.1
        suc=False
        
        while suc == False:
            print("AAA")

            j=random.randint(0,int(m))
            cmm=np.arange(R*L+1)
            c=initialize()
            for i in range(j):
                c=Groove(c)
            #print(c._nqubits)
            r=c.measure(0,1,2,3,4,with_prob=False)
            cmm=K.numpy(r[0])
            
            sum=0
            cmminput=np.zeros([2**(R*L+1)])
            for i in range(R*L):
                sum=sum+(cmm[i]*2**i)
            cmminput[int(sum)]=1
            cmmm=tc.Circuit(R*L+1,inputs=cmminput)
            if classy(cmmm) == 1:
                suc == True
                return np.delete(cmm,[-1])
            m = min(m*m_,math.sqrt(2**(R*L)))

print(mainloop())
            