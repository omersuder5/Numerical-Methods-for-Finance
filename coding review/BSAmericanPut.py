import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.stats import norm


import PSOR  # For this line to work, PSOR.py must be in the same directory as this script.


PSOR.testPSOR()  # Calls testPSOR() which is defined in PSOR.py.


R = 3
leftLim = -R / 2
rightLim = R / 2


# Explicit B-S formulas:
def bs_formula_P(r, sigma, s, t, K):
    # Value of an European put option
    if np.isclose(t, 0):
        t = 1e-10  # Replace by very small value to avoid runtime warning
    d1 = (np.log(s / K) + (0.5 * sigma**2 + r) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    P = -s * norm.cdf(-d1) + K * np.exp(-r * t) * norm.cdf(-d2)
    return P


# FEM assembling routines, to be completed for 3c)
def build_massMatrix(N,R):
    # todo 3 a)
    h=2*R/(N+1)
    a=2/3*np.ones(N)
    b=1/6*np.ones(N-1)
    M = sp.diags(a,0)+sp.diags(b,1)+sp.diags(b,-1)
    return M*h

def build_rigidityMatrix(N,R):
    # todo 3 a)
    h=2*R/(N+1)
    a=2/h*np.ones(N)
    b=-1/h*np.ones(N-1)
    A = sp.diags(a,0)+sp.diags(b,1)+sp.diags(b,-1)
    return A

def build_W(N):
    return 1 / 2 * tridiag(N, -1, 0, 1)


def buildBS_matrix(N,R,sigma,r):
    A = build_rigidityMatrix(N,R)
    M = build_massMatrix(N,R)
    W = build_W(N)
    return 1/2*sigma**A +r*M + (r-1/2*sigma**2)*W

def build_G(N, r, sigma, K):
    R=3
    h=2*R/(N+1)
    x=np.array([-R+i*h for i in range(1,N+1)])
    ind1 = np.array(x(x<=np.log(K)))

# Solver, to be completed for 3c)


def computeOptionValue(r, sigma, K, T, N, M):
    R=3
    h=2*R/(N+1)
    x=np.array([-R+i*h for i in range(N+2)])
    k=T/M
    M=build_massMatrix(N,R)
    A=build_rigidityMatrix(N,R)
    B=M+k*A
    
    return


# Plotting routine
def plotFE(vec, lab):
    N = vec.size
    xi = np.linspace(-R, R, N + 2)
    vals = 0 * xi
    vals[1:-1] = vec
    ind = (xi > leftLim) * (xi < rightLim)
    plt.plot(np.exp(xi[ind]), vals[ind], label=lab)


# Black-Scholes model parameters:

r = 0.05  # Riskless interest rate
sigma = 0.3  # Underlying volatility
K = 1  # Strike price
T = 1  # Maturity
N = 401  # Grid size
M = 400  # Number of time steps


def gput(x):
    S = np.exp(x)
    val = np.maximum(K - S, 0)
    return val


# Computation of the option value
Vamput = computeOptionValue(r, sigma, K, T, N, M)

# Visualization
xi = np.linspace(-R, R, N + 2)
S = np.exp(xi)
ind = (xi > leftLim) * (xi < rightLim)
plotFE(Vamput, "American put")
plt.plot(S[ind], gput(xi[ind]), "k--", label="payoff")
plt.plot(
    S[ind],
    bs_formula_P(r=r, sigma=sigma, s=S[ind], t=T, K=K),
    "r--",
    label="European put",
)
plt.legend()
plt.xlabel("Stock price s")
plt.ylabel("Option value")
plt.title("American vs European put option")
plt.show()
