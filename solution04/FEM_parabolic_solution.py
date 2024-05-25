import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy.linalg as lin


def build_massMatrix(N):
    a = 1 / 6 * np.ones(N - 1)
    M = sp.diags(a, -1) + sp.diags(a, 1) + 2 / 3 * sp.eye(N)
    return 1 / (N + 1) * M


def alpha(x):
    return 1 + x * x


def beta(x):
    return 2 * x


def gamma(x):
    return np.pi**2 * x**2


########### Alternatively could use an element-by-element construction ###########
# def build_rigidityMatrix(N,alpha,beta,gamma):
#     M=np.zeros((N,N))
#     h=1/(N+1)
#     k=h/2 #Assume the mapping takes the form x=hatx*k+b, this line computes k
#     for i in range(N+1):
#         b=i*h+h/2 #This line computes b
#         if i==0:
#            M[0,0]=M[0,0] + 1/3*( (2/h*alpha(-k+b)/4) + 4*(2/h*alpha(b)/4+beta(b)/4+h/2*gamma(b)/4) + (2/h*alpha(k+b)/4+beta(k+b)/2+h/2*gamma(k+b)) )
#         elif i==N:
#            M[N-1,N-1]=M[N-1,N-1] + 1/3*((2/h*alpha(-k+b)/4-beta(-k+b)/2+h/2*gamma(-k+b)) + 4*(2/h*alpha(b)/4-beta(b)/4+h/2*gamma(b)/4) + (2/h*alpha(k+b)/4))
#         else:
#            M[i-1,i-1]=M[i-1,i-1] + 1/3*((2/h*alpha(-k+b)/4-beta(-k+b)/2+h/2*gamma(-k+b)) + 4*(2/h*alpha(b)/4-beta(b)/4+h/2*gamma(b)/4) + (2/h*alpha(k+b)/4))
#            M[i,i]=M[i,i] + 1/3*( (2/h*alpha(-k+b)/4) + 4*(2/h*alpha(b)/4+beta(b)/4+h/2*gamma(b)/4) + (2/h*alpha(k+b)/4+beta(k+b)/2+h/2*gamma(k+b)) )
#            M[i-1,i]=M[i-1,i] + 1/3*((-2/h*alpha(-k+b)/4+beta(-k+b)/2) + 4*(-2/h*alpha(b)/4+beta(b)/4+h/2*gamma(b)/4) + (-2/h*alpha(k+b)/4))
#            M[i,i-1]=M[i,i-1]+1/3*((-2/h*alpha(-k+b)/4) + 4*(-2/h*alpha(b)/4-beta(b)/4+h/2*gamma(b)/4) + (-2/h*alpha(k+b)/4-beta(k+b)/2))
#     return M


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
        +2*h*h*c2[1:-1]
    )/6

    diag_down = 1/h*(
        -a1[1:-2]-4*a2[1:-1]-a1[2:-1]
        -2*h*b2[1:-1]-h*b1[2:-1]
        +2*h*h*c2[1:-1]
    )/6

    return sp.diags([diag_down, diag0, diag_up], [-1, 0, 1])


def f(t, x):
    return np.exp(-t) * (2 * np.pi**2 * x**2 + np.pi**2 - 1) * np.sin(np.pi * x)


def initial_value(x):
    return np.sin(np.pi * x)


def exact_solution_at_1(x):
    return np.exp(-1) * np.sin(np.pi * x)


def build_F(t, N):
    h = 1 / (N + 1)
    X = np.linspace(0, 1, N + 2)
    return h / 3 * (f(t, X[:-2] + h / 2) + f(t, X[1:-1]) + f(t, X[1:-1] + h / 2))


def FEM_theta(N, M, theta):
    k = 1 / M
    grid = (1 / (N + 1)) * np.arange(1, N + 1)
    u_sol = initial_value(grid)
    MatrixM = build_massMatrix(N)
    MatrixA = build_rigidityMatrix(N, alpha, beta, gamma)
    B_theta = MatrixM + k * theta * MatrixA
    C_theta = MatrixM - k * (1 - theta) * MatrixA

    B_theta = B_theta.tocsr()
    C_theta = C_theta.tocsr()

    for i in range(M):
        F_theta = k * theta * build_F(k * (i + 1), N) + k * (1 - theta) * build_F(
            k * i, N
        )
        RHS = C_theta @ u_sol + F_theta
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
