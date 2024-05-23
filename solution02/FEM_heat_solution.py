import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def build_massMatrix(N):
    a = 1/6 * np.ones(N-1)
    M = sp.diags(a,-1) + sp.diags(a,1) + 2/3 * sp.eye(N)
    return 1/(N+1) * M


def build_rigidityMatrix(N):
    a = -1 * np.ones(N-1)
    M = sp.diags(a,-1) + sp.diags(a,1) + 2*sp.eye(N)
    return (N+1) * M


def f(t,x):
    return np.exp(-t)*((np.pi**2-1)*x*np.sin(np.pi*x)-2*np.pi*np.cos(np.pi*x))


def initial_value(x):
    return x*np.sin(np.pi*x)


def exact_solution_at_1(x):
    return np.exp(-1)*x*np.sin(np.pi*x)


def build_F(t,N):
    h = 1/(N+1)
    i_values = np.arange(N)
    return h * 1/3 * (f(t, h * (i_values + 0.5)) + f(t, h * (i_values + 1)) + f(t, h * (i_values + 1.5)))


def FEM_theta(N,M,theta):
    k = 1/M
    grid = (1/(N+1))*(np.arange(N)+1)
    u_sol = initial_value(grid)
    MatrixM = build_massMatrix(N)
    MatrixA = build_rigidityMatrix(N)
    B_theta = MatrixM + k*theta*MatrixA
    C_theta = MatrixM - k*(1-theta)*MatrixA
    
    B_theta = B_theta.tocsr() 
    C_theta = C_theta.tocsr()
    
    for i in range(M):
        F_theta = k*theta*build_F(k*(i+1),N) + k*(1-theta)*build_F(k*i,N)
        RHS = C_theta@u_sol + F_theta
        u_sol = spsolve(B_theta, RHS)
        
    return u_sol


#### error analysis ####
nb_samples = 5
N = np.power(2, np.arange(2, 2 + nb_samples))-1
# M in the case of 3 f)
M = np.power(4, np.arange(2, 2 + nb_samples))
# M in the case of 3 e)
# M = np.power(2, np.arange(2, 2 + nb_samples))
theta=1 #Change the theta to 0.5 or 1 when needed in 3 e)-f)


#### Do not change any code below! ####
l2error = np.zeros(nb_samples) 
k =  1 / M

try:
   for i in range(nb_samples):
      l2error[i] = (1 / (N[i]+1)) ** (1 / 2) * lin.norm(exact_solution_at_1((1/(N[i]+1))*(np.arange(N[i])+1)) - FEM_theta(N[i], M[i],theta), ord=2)
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
