import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def build_massMatrix(N, beta):
    h = np.diff(np.linspace(0,1,N+2)**beta)
    return sp.diags((h[:-1] + h[1:])/3, 0) + sp.diags(h[1:-1]/6, 1) + sp.diags(h[1:-1]/6, -1)

def build_rigidityMatrix(N, beta):
    h = np.diff(np.linspace(0,1,N+2)**beta)
    return sp.diags(1/h[:-1] + 1/h[1:], 0) + sp.diags(-1/h[1:-1], 1) + sp.diags(-1/h[1:-1], -1)

def f(t,x):
    return np.exp(-t)*((np.pi**2-1)*x*np.sin(np.pi*x)-2*np.pi*np.cos(np.pi*x))

def initial_value(x):
    return x*np.sin(np.pi*x)


def exact_solution_at_1(x):
    return np.exp(-1)*x*np.sin(np.pi*x)


def build_F(t, N, beta):
    X = np.linspace(0,1,N+2)**beta
    h = np.diff(X)
    a = f(t, X[1:-1])
    ab2 = f(t, (X[:-2] + X[1:-1])/2)
    b = f(t, (X[1:-1] + X[2:])/2)
    
    return a*(h[:-1] + h[1:])/6 + ab2*h[:-1]/3 + b*h[1:]/3


def FEM_theta(N, M, beta, theta):
    k = 1/M
    u_sol = initial_value((np.linspace(0,1,N+2)**beta)[1:-1])
    
    MatrixM = build_massMatrix(N, beta)
    MatrixA = build_rigidityMatrix(N, beta)
    B_theta = MatrixM + k*theta*MatrixA
    C_theta = MatrixM - k*(1-theta)*MatrixA
    
    B_theta = B_theta.tocsr() 
    C_theta = C_theta.tocsr()

    for i in range(M):
        F_theta = k*theta*build_F(k*(i+1), N, beta) + k*(1-theta)*build_F(k*i, N, beta)
        RHS = C_theta@u_sol + F_theta
        u_sol = spsolve(B_theta, RHS)
        
    return u_sol


#### error analysis ####
nb_samples = 5
N = np.power(2, np.arange(2, 2 + nb_samples))-1
# M in the case of 3 d)
# M = np.power(4, np.arange(2, 2 + nb_samples))
# M in the case of 3 e)
M = 7*np.power(4, np.arange(2, 2 + nb_samples))
theta = 0
beta = 1.05

#### Do not change any code below! ####
l2error = np.zeros(nb_samples)
k =  1/M

try:
    for i in range(nb_samples):
        h = np.diff(np.linspace(0,1,N[i]+2)**beta)[:-1]
        l2error[i] = lin.norm((exact_solution_at_1((np.linspace(0,1,N[i]+2)**beta)[1:-1]) - FEM_theta(N[i], M[i], beta, theta))*np.sqrt(h))
        if np.isnan(l2error[i])==True:
              raise Exception("Error unbounded. Plots not shown.")
    
    conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
    if conv_rate[0]<0:
       raise Exception("Error unbounded. Plots not shown.")
    print(f"FEM method with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}")
    plt.figure(figsize=[10, 6])
    plt.loglog(k, l2error, '-x', label='error')
    plt.loglog(k, k, '--', label='$O(k)$')
    plt.loglog(k, k**2, '--', label='$O(k^2)$')
    plt.title('$L^2$ convergence rate', fontsize=13)
    plt.xlabel('$k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
    plt.show()
except Exception as e:
    print(e)