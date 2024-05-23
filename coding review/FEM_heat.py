import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def build_massMatrix(N):
    # todo 3 a)
    h=1/(N+1)
    a=2/3*np.ones(N)
    b=1/6*np.ones(N-1)
    M = sp.diags(a,0)+sp.diags(b,1)+sp.diags(b,-1)
    return M*h


def build_rigidityMatrix(N):
    # todo 3 a)
    h=1/(N+1)
    a=2/h*np.ones(N)
    b=-1/h*np.ones(N-1)
    A = sp.diags(a,0)+sp.diags(b,1)+sp.diags(b,-1)
    return A


def f(t,x):
    # todo 3 b)
    return np.exp(-t)*((np.pi**2-1)*x*np.sin(np.pi*x)-2*np.pi*np.cos(np.pi*x))
    

def initial_value(x):
    # todo 3 b)
    return x*np.sin(np.pi*x)


def exact_solution_at_1(x):
    # todo 3 b)
    return np.exp(-1)*x*np.sin(np.pi*x)


def build_F(t,N):
    # todo 3 c)
    h=1/(N+1)
    x_i=[h*(i+1) for i in range(N)] 
    x_i=np.array(x_i)
    return h/3*(f(t,x_i-h/2)+f(t,x_i)+f(t,x_i+h/2))


def FEM_theta(N,M,theta):
    # todo 3 d)
    k=1/M
    h=1/(N+1)
    x_i=[h*(i+1) for i in range(N)] 
    x_i=np.array(x_i)
    u_sol=initial_value(x_i)

    MatrixM=build_massMatrix(N)
    MatrixA=build_rigidityMatrix(N)
    B_theta=MatrixM+k*theta*MatrixA
    C_theta=MatrixM-k*(1-theta)*MatrixA

    B_theta=B_theta.tocsr()
    C_theta=C_theta.tocsr()

    for m in range(M):
        F_theta= k*theta*build_F(k*(m+1),N)+k*(1-theta)*build_F(k*m,N)
        u_sol=sp.linalg.spsolve(B_theta,C_theta@u_sol+F_theta)
    return u_sol


#### error analysis ####
nb_samples = 5
N = [2**l-1 for l in range(2,7)] # todo for 3 c)
M = [4**l for l in range(2,7)] # todo  for 3 c) and 3 d)
#theta= 0.3
#theta = 0.5
theta = 1

M=np.array(M)
N=np.array(N)
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



