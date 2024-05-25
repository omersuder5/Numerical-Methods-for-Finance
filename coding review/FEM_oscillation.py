import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import numpy.linalg as lin


def build_massMatrix(N):
    a = 1 / 6 * np.ones(N - 1)
    M = sp.diags(a, -1) + sp.diags(a, 1) + 2 / 3 * sp.eye(N)
    return 1 / (N + 1) * M


def build_rigidityMatrix(N):
    a = -1 * np.ones(N - 1)
    M = sp.diags(a, -1) + sp.diags(a, 1) + 2 * sp.eye(N)
    return (N + 1) * M


def f(t, x):
    return x * t


def initial_value(x):
    return ((x >= 0.25) * (x <= 0.75)).astype(float)


def build_F(t, N):
    X = np.linspace(0, 1, N + 2)
    h = np.diff(X)
    a = f(t, X[1:-1])
    ab2 = f(t, (X[:-2] + X[1:-1]) / 2)
    b = f(t, (X[1:-1] + X[2:]) / 2)

    return a * (h[:-1] + h[1:]) / 6 + ab2 * h[:-1] / 3 + b * h[1:] / 3


def FEM_theta(N, M, theta, beta):
    # implement the theta scheme for time step t_j = (j/M)^beta  
    t = np.array([(j/M)**beta for j in range(M+1)])
    k = np.diff(t)
    A_matrix = build_rigidityMatrix(N).tocsr()
    M_matrix = build_massMatrix(N).tocsr()
    x = np.array([1/(N+1)*(i+1) for i in range(N)])
    u = initial_value(x)

    # now k is not constant, need compute the matrices in each iteration
    for m in range(M):
        F= k[m]*theta * build_F(t[m+1], N) + k[m]*(1-theta)*build_F(t[m], N)
        B = M_matrix + k[m]*theta*A_matrix
        C = M_matrix - k[m]*(1-theta)*A_matrix
        u = spsolve(B, C@u + F)
    return u


#### error analysis ####
nb_samples = 3
N = np.power(2, np.arange(8, 8 + nb_samples)) - 1
M = np.power(2, np.arange(8, 8 + nb_samples))
theta = 0.5
beta = 15# set beta according to b) and d)

step_size1 = 1/(N[0]+1)
step_size2 = 1/(N[1]+1)

uh1 = FEM_theta(N[0], M[0], theta, beta)
uh2 = FEM_theta(N[1], M[1], theta, beta)
uh3 = FEM_theta(N[2], M[2], theta, beta)
conv_rate = np.log(
    np.sqrt(step_size1*np.sum((uh1-uh2[1::2])**2))
    /np.sqrt(step_size2*np.sum((uh2-uh3[1::2])**2))
)/np.log(2)


print(
    f"FEM with theta={theta}: Convergence rate in discrete l^2 norm with respect to time step $k$: {conv_rate}"
)