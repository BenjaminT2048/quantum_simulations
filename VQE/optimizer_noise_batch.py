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
level = 5
batch = 10
px = 0.1
py = 0.1
pz = 0.1
h_range = 10
def min(h):
    h=h
    def rzz(c,i,j, params):
        c.cnot(i,i+1)
        c.rz(i+1,theta=params[4*j+1, i])
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
    

    def ex(params, seed):
        c=tc.Circuit(qubit)
        k=level

        for i in range(qubit):
            c.h(i)
        for j in range(k):
            for i in range(qubit):
                c.ry(i,theta=params[4*j+3,i])
                c.rz(i,theta=params[4*j+2,i])
                c.s(i)
            for i in range(qubit):
                c.rx(i,theta=params[4*j, i])
            for i in range(qubit-1):
                c = rzz(c,i,j, params)
            for i in range(qubit):
                c.depolarizing(i, px=px, py=py, pz=pz, status=seed[j][i])
        return energy(c)

    ex_vg = tc.backend.jit(tc.backend.vvag(ex, argnums=0, vectorized_argnums=(0,1)))

    P = tf.Variable(initial_value=tf.random.normal(shape=[batch, level * 4+3, qubit], stddev=0.1, dtype=getattr(tf, tc.rdtypestr)))
    params = P
    seed = K.ones(shape = [batch, level, qubit], dtype = "float32")
    history = [ ]
    opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))

    #t_2=time.time()
    v_0=0
    for _ in range(200):
        v, g = ex_vg(params, seed)
        params = opt.update(g, params)
    #     if _ % 20 == 0:
    #         t_2=time.time()
    #         print(v)
    #         print(t_2-t_1,"s")
    #         t_1=t_2
        if _ == 199:
            v_0=np.min(v.numpy())

    #plt.plot([i for i in range(1000)], history)
    #plt.ylabel("value")
    #plt.xlabel("training step")
    # print((t_2-t_0)/60,"m")
    #print(history[-1])
    return v_0
x = np.arange(h_range)
y = np.zeros(h_range, dtype="float32")
for i in range(h_range):
    x[i]=0.1*i
    y[i]=min(x[i])
    print(y[i])
plt.plot(x,y)
plt.show()