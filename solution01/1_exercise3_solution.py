import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp

# Set how floating-point errors are handled.
np.seterr(all='raise')


def initial_value(x):
    return np.sin(np.pi/2 * x)


#### exact solution at t=1 ####
def exact_solution_at_1(x):
    return np.exp(-np.pi**2 / 4) * np.sin(np.pi/2 * x)


#### numerical scheme ####
def eulerexplicit(N, M):
    h = 1 / N  # x-direction step
    k = 1 / M  # t-direction step
    a = -2 * np.ones(N)
    b = np.ones(N-1)
    b[-1] = 2
    c = np.ones(N-1)
    G = sp.diags(a, 0) + sp.diags(b, -1) + sp.diags(c, 1)
    C = sp.eye(N) + k / (h**2) * G

    u_M = initial_value(np.linspace(0, 1, N + 1))[1:] # remove u_0

    for _ in range(M):
        u_M = C@u_M
    return u_M


def eulerimplicit(N, M):
    h = 1 / N  # x-direction step
    k = 1 / M  # t-direction step
    a = - 2 * np.ones(N)
    b = np.ones(N-1)
    b[-1] = 2
    c = np.ones(N-1)
    G = sp.diags(a, 0) + sp.diags(b, -1) + sp.diags(c, 1)
    C = sp.eye(N) - k / (h**2) * G

    u_M = initial_value(np.linspace(0, 1, N + 1))[1:] # remove u_0
    C = C.tocsr() # convert to csr or csc format for scipy.linalg.spsolve()

    for i in range(M):
        u_M = sp.linalg.spsolve(C, u_M)
    return u_M


#### error analysis ####
nb_samples = 5
N = np.power(2, np.arange(2, 2 + nb_samples))
# M in the case of 3 c)
# M = 2 * np.power(4, np.arange(2, 2 + nb_samples))
# M in the case of  3 d)
M = 2*np.power(4, np.arange(2, 2 + nb_samples))
l2errorexplicit = np.zeros(nb_samples)  # error vector for explicit method
l2errorimplicit = np.zeros(nb_samples)  # error vector for implicit method
h2k = 1 / (N ** 2) + 1 / M


#### Do not change any code below! ####
try:
    for i in range(nb_samples):
        l2errorexplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerexplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorexplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for explicit method. Plots not shown.")
    print(f"Explicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: {conv_rate[0]:.4f}")
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
    print(f"Implicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: {conv_rate[0]:.4f}")
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