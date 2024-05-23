import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Set how floating-point errors are handled.
np.seterr(all='raise')


def initial_value(x):
    return np.sin(np.pi/2 * x)


#### exact solution at t=1 ####
def exact_solution_at_1(x):
    return np.sin(np.pi/2 * x)*np.exp(-np.pi**2/4)


#### numerical scheme ####
def eulerexplicit(N, M):
    # todo 3 b)
    k=1/M
    h=1/N
    ones= np.ones(N-1)
    a=sp.diags(1*ones,1)
    b=sp.diags(-2*np.ones(N),0)
    ones[-1]=2
    c=sp.diags(1*ones,-1)
    G = a+b+c
    C=sp.eye(N)+k/h**2*G
    u=initial_value(np.linspace(0,1,N+1))[1:] # drop the first element
    for m in range(M):
        u=C.dot(u)
    return u


def eulerimplicit(N, M):
    # todo 3 b)
    k=1/M
    h=1/N
    ones= np.ones(N-1)
    a=sp.diags(1*ones,1)
    b=sp.diags(-2*np.ones(N),0)
    ones[-1]=2
    c=sp.diags(1*ones,-1)
    G = a+b+c
    C=sp.eye(N)-k/h**2*G
    u=initial_value(np.linspace(0,1,N+1))[1:] # drop the first element
    for m in range(M):
        u=spsolve(C,u)
    return u


#### error analysis ####
nb_samples = 5
N = [2**i for i in range(2,7)]
M = [2*4**i for i in range(2,7)]
l2errorexplicit = np.zeros(nb_samples)  # error vector for explicit method
l2errorimplicit = np.zeros(nb_samples)  # error vector for implicit method
#h2k = 1 / (N ** 2) + 1 / M
h2k = [1/(N[i]**2)+1/M[i] for i in range(len(N))]


#### Do not change any code below! ####
try:
    for i in range(nb_samples):
        l2errorexplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerexplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorexplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for explicit method. Plots not shown.")
    print("Explicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorexplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='$O(h^2+k)$')
    plt.title('$L^2$ convergence rate for explicit method', fontsize=13)
    plt.xlabel('$h^2+k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print(f"Exception: {e}")

try:
    for i in range(nb_samples):
        l2errorimplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerimplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorimplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for implicit method. Plots not shown.")
    print("Implicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorimplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='$O(h^2+k)$')
    plt.title('$L^2$ convergence rate for implicit method', fontsize=13)
    plt.xlabel('$h^2+k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print(f"Exception: {e}")

plt.show()