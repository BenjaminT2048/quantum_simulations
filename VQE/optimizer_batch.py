#import time
#t_0=time.time()

from functools import partial
from re import X
import numpy as np
import tensorflow as tf
#import jax
#from jax.config import config

#config.update("jax_enable_x64", True)
#from jax import numpy as jnp
#from jax.experimental import optimizers
import tensorcircuit as tc
import math
import matplotlib.pyplot as plt

#t_1=time.time()
#print(t_1-t_0,"s")

K = tc.set_backend("tensorflow")
zz = np.kron(tc.gates._z_matrix, tc.gates._z_matrix)

qubit = 5
level = 10
batch = 2
def min(h):
    def rzz(c,i,j, params):
        c.cnot(i,i+1)
        c.rz(i+1,theta=params[j, 0])
        c.cnot(i,i+1)
        return c

    def energy(c: tc.Circuit):
        e = 0.0
        n = c._nqubits
        for i in range(n):
            e += h * c.expectation((tc.gates.x(), [i]))  # <X_i>
        for i in range(n - 1):  # OBC
            e += (-1.0) * c.expectation(
                (tc.gates.z(), [i]), (tc.gates.z(), [(i + 1) % n])
            )  # <Z_iZ_{i+1}>
        return tc.backend.real(e)
    

    def ex(params):
        c=tc.Circuit(qubit)
        k=level
        for i in range(qubit):
            c.h(i)
        for j in range(k):
            for i in range(qubit):
                c.rx(i,theta=params[j, 1])
            for i in range(qubit-1):
                c = rzz(c,i,j, params)
        return energy(c)

    ex_vg = tc.backend.jit(tc.backend.vvag(ex, argnums=0, vectorized_argnums=0))

    P = tf.Variable(initial_value=tf.random.normal(shape=[batch, level * 2, qubit], stddev=0.1, dtype=getattr(tf, tc.rdtypestr)))
    params = P
    history = [ ]
    opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))

    #t_2=time.time()
    v_0=0
    for _ in range(200):
        v, g = ex_vg(params)
        params = opt.update(g, params)
        #if _ % 20 == 0:
            #t_2=time.time()
            #print(v)
            #print(t_2-t_1,"s")
            #t_1=t_2
        #history.append(np.min(v.numpy()))
        if _ == 200:
            v_0=np.min(v.numpy())
    return v_0
    #plt.plot([i for i in range(100)], history)
    #plt.ylabel("infidelity")
    #plt.xlabel("training step")
    #print((t_2-t_0)/60,"m")
    #print(history[-1])
x=np.arange(20)
y=x
for i in range(20):
    x[i]=0.2*i
    y[i]=min(x[i])
plt.plot(x,y)
plt.show()