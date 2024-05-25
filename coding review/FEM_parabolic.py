import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy.linalg as lin


def build_massMatrix(N):
    # Todo : Implement the function build_massMatrix
    h=1/(N+1)
    a=sp.diags(2/3*h*np.ones(N),0)
    b=sp.diags(1/6*h*np.ones(N-1),-1)
    c=sp.diags(1/6*h*np.ones(N-1),1)
    return a+b+c


def alpha(x):
    # Todo : Implement the function alpha
    return 1+x**2


def beta(x):
    # Todo : Implement the function beta
    return 2*x


def gamma(x):
    # Todo : Implement the function gamma
    return np.pi**2*x**2 


def build_rigidityMatrix(N, alpha, beta, gamma):
    # Todo : Implement the function build_rigidityMatrix
    h=1/(N+1)
    x = np.array([i*h for i in range(N+2)])
    y = (x[1:]+x[:-1])/2

    a1 = alpha(x)
    a2 = alpha(y)
    b1 = beta(x)
    b2 = beta(y)
    c1 = gamma(x)
    c2 = gamma(y)

    diag0 = 1/h*(
        a1[:-2]+4*a2[:-1]+a1[1:-1]
        +2*h*b2[:-1]+h*b1[1:-1]
        +h*h*c1[1:-1]+h*h*c2[:-1]

        +a1[1:-1]+4*a2[1:]+a1[2:]
        -2*h*b2[1:]-h*b1[1:-1]
        +h*h*c1[1:-1]+h*h*c2[1:]
    )/6
    
    diag_up = 1/h*(
        -a1[1:-2]-4*a2[1:-1]-a1[2:-1]
        +2*h*b2[1:-1]+h*b1[1:-2]
        +h*h*c2[1:-1]
    )/6

    diag_down = 1/h*(
        -a1[1:-2]-4*a2[1:-1]-a1[2:-1]
        -2*h*b2[1:-1]-h*b1[2:-1]
        +h*h*c2[1:-1]
    )/6

    return sp.diags([diag_down, diag0, diag_up], [-1, 0, 1])

def f(t, x):
    # Todo : Implement the function f
    return (2*np.pi**2*x**2+np.pi**2-1)*np.exp(-t)*np.sin(np.pi*x)


def initial_value(x):
    # Todo : Implement the function initial_value
    return np.sin(np.pi*x)


def exact_solution_at_1(x):
    # Todo : Implement the function exact_solution_at_1
    return np.exp(-1)*np.sin(np.pi*x)


def build_F(t,N):
    # todo 3 c)
    h=1/(N+1)
    x=np.array([h*i for i in range(N+2)])
    return h/3*(f(t,x[1:-1]-h/2)+f(t,x[1:-1])+f(t,x[1:-1]+h/2))

def FEM_theta(N, M, theta):
    # Todo : Implement the theta scheme, return the solution u_sol at final time
    h = 1/(N+1)
    k = 1/M
    x = np.array([i*h for i in range(1,N+1)])
    u_sol = initial_value(x)
    MatrixM = build_massMatrix(N)
    MatrixA = build_rigidityMatrix(N, alpha, beta, gamma)
    B_theta = MatrixM + k * theta * MatrixA
    C_theta = MatrixM - k * (1 - theta) * MatrixA

    B_theta = B_theta.tocsr()
    C_theta = C_theta.tocsr()

    for m in range(M):
        F_theta = k*theta*build_F(k*(m+1),N) + k*(1-theta)*build_F(k*m,N)
        RHS = C_theta*u_sol + F_theta
        u_sol = spsolve(B_theta, RHS)

    return u_sol


#### error analysis ####
nb_samples = 5
N = np.power(2, np.arange(5, 5 + nb_samples)) - 1
M = np.power(2, np.arange(5, 5 + nb_samples))
theta = 0.5

#### Do not change any code below! ####
l2error = np.zeros(nb_samples)
k = 1 / M


try:
    for i in range(nb_samples):
        l2error[i] = (1 / (N[i] + 1)) ** (1 / 2) * lin.norm(
            exact_solution_at_1((1 / (N[i] + 1)) * (np.arange(N[i]) + 1))
            - FEM_theta(N[i], M[i], theta),
            ord=2,
        )
        if np.isnan(l2error[i]) == True:
            raise Exception("Error unbounded. Plots not shown.")
    conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
    if conv_rate[0] < 0:
        raise Exception("Error unbounded. Plots not shown.")
    print(
        f"FEM with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}"
    )
    plt.figure(figsize=[10, 6])
    plt.loglog(k, l2error, "-x", label="error")
    plt.loglog(k, k, "--", label="$O(k)$")
    plt.loglog(k, k**2, "--", label="$O(k^2)$")
    plt.title("$L^2$ convergence rate", fontsize=13)
    plt.xlabel("$k$", fontsize=13)
    plt.ylabel("error", fontsize=13)
    plt.legend()
    plt.plot()
    plt.show()
except Exception as e:
    print(e)
