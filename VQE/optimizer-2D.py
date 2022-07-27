#import time
#t_0=time.time()

from functools import partial
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
R=2
L=2
qubit = R*L
level = 4
batch = 1
h = [5 for i in range(qubit)]


def rzz(c,i,j, params):
    c.cnot(i,i+1)
    c.rz(i+1,theta=params[4*j+1, i])
    c.cnot(i,i+1)
    return c

def energy(c: tc.Circuit,R,L):
    e = 0.0
    n = c._nqubits

    for i in range(n):
        e += h[i] * c.expectation((tc.gates.x(), [i]))  # <X_i>
    for i in range(R):
        for j in range(L-1):
            e += (-1.0) * c.expectation((tc.gates.z(), [j*R+i]), (tc.gates.z(), [(j+1)*R+i]))
    for j in range(L):
        for i in range(R-1):
            e += (-1.0) * c.expectation((tc.gates.z(), [j*R+i]), (tc.gates.z(), [j*R+i+1]))
    return tc.backend.real(e)
    

def ex(params):
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
    return energy(c,R,L)

ex_vg = tc.backend.jit(tc.backend.vvag(ex, argnums=0, vectorized_argnums=0))

P = tf.Variable(initial_value=tf.random.normal(shape=[batch, level * 4+3, qubit], stddev=0.1, dtype=getattr(tf, tc.rdtypestr)))
params = P
history = [ ]
opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))

#t_2=time.time()

for _ in range(100):
    v, g = ex_vg(params)
    params = opt.update(g, params)
#     if _ % 20 == 0:
#         t_2=time.time()
#         print(v)
#         print(t_2-t_1,"s")
#         t_1=t_2
    history.append(np.min(v.numpy()))

plt.plot([i for i in range(100)], history)
plt.ylabel("value")
plt.xlabel("training step")
plt.plot()
print(history[-1])
