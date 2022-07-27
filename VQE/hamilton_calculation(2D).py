import tensorcircuit as tc
import tensorflow as ts 
import numpy as np
import math
from numpy import linalg as LA

tc.set_backend("tensorflow")

def i_iterate_multiplication(n):
    tmp = 1
    for i in range(n):
        tmp = np.kron(I, tmp)
    return tmp
R=2
L=2
h=5
n = R*L
# h_i = 1
H = 0
theta = math.pi/2
min_w = 999
I = tc.gates._i_matrix
X = tc.gates._x_matrix  
Y = tc.gates._y_matrix  
Z = tc.gates._z_matrix 


for j in range(L):
    for i in range(R-1):
        H = H - np.kron(np.kron(i_iterate_multiplication(j*R+i), np.kron(Z, Z)), i_iterate_multiplication(n-2-j*R-i))
for i in range(R):
    for j in range(L-1):
        H = H - np.kron(np.kron(i_iterate_multiplication(j*R+i), np.kron(Z,np.kron(i_iterate_multiplication(R-1),Z))), i_iterate_multiplication(n-1-(j+1)*R-i))
for i in range(0, n):
    H = H + h * (np.kron(np.kron(i_iterate_multiplication(i), X), i_iterate_multiplication(n-1-i)))
# print(H)

w, v = LA.eig(H)

for i in range(0, len(w)):
    if w[i] < min_w:
        min_w = w[i]
print(h)
print(min_w)