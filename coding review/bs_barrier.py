import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

def bseucall(S, T, K, r, sigma):
    """
    Computes the value of a European put option with the Black-Scholes model using analytic formulas.

    Parameters:
        S (float or array-like): Stock prices at time 0.
        T (float): Maturity.
        K (float): Strike.
        r (float): Interest rate.
        sigma (float): Volatility.

    Returns:
        P (float or array-like): Option price at time 0.
    """
    # call option
    d1 = (np.log(S / K) + (0.5 * sigma ** 2 + r) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    P = S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
    return P

# Compute Generator/ Source Term / Initial Data
def initial_value(x):
    u_0 = np.maximum(np.exp(x) - 1, 0)
    return u_0

def buildMassBS(N, h):
    a = h/6 * np.ones(N-1)
    b = 2 * h/3 * np.ones(N)
    M = sp.diags([a, b, a], [-1, 0, 1])
    return M

def buildABS(N, h, sigma, r):
    As = 1/h * diags([-np.ones(N-1), 2 * np.ones(N), -np.ones(N-1)], [-1, 0, 1])
    Ab = 1/2 * diags([-np.ones(N-1), np.zeros(N), np.ones(N-1)], [-1, 0, 1])
    Am = buildMassBS(N, h)
    return 1/2*sigma**2*As + (sigma**2/2-r)*Ab + r*Am


# Solver
def FEM_theta(N, M, R, B, r, sigma, K, T, theta):
    h = (np.log(B/K) + R) / (N + 1)
    x = np.array([-R + i*h for i in range(1, N+1)])
    u_sol = initial_value(x)
    MatrixM = buildMassBS(N, h)
    MatrixA = buildABS(N, h, sigma, r)
    k = T / M
    B_theta = (MatrixM + k * theta * MatrixA).tocsr()
    C_theta = MatrixM - k * (1 - theta) * MatrixA
    for m in range(M):
        u_sol=sp.linalg.spsolve(B_theta, C_theta @ u_sol)
    
    return u_sol

if __name__ == '__main__':
    # Set Parameters
    N = 255                     # number of nodes
    M = 256                     # number of time steps
    R = 3                       # localization
    T = 1                       # maturity
    K = 60                      # strike
    r = 0.01                    # interest rate
    sigma = 0.3                 # volatility
    B = 80                      # barrier

    
    # compute up-and-out barrier
    theta=0.5
    uout = FEM_theta(N, M, R, B, r, sigma, K, T, theta)
    uout = K*uout # scale back to the original problem

    # compute up-and-in barrier
    h = (np.log(B/K) + R) / (N + 1)
    x = np.array([-R + i*h for i in range(1, N+1)])
    S = np.exp(x)*K   #needed to compute the bseucall
    ubs = bseucall(S, T, K, r, sigma)
    uin = ubs-uout

    # Postprocessing
    # area of interest
    x = np.linspace(-R, np.log(B/K), N + 2)[1:-1]
    S = np.exp(x)*K
    I = np.abs(x) < 0.75


    # plot solution
    plt.figure(1)
    plt.plot(S[I], uout[I], 'bx-', label='Knock-out barrier', markersize=2, linewidth=0.5)
    plt.plot(S[I], uin[I], 'go-', label='Knock-in barrier', markersize=2, linewidth=0.5)
    plt.plot(S[I], ubs[I], 'rs-', label='Plain vanilla', markersize=2, linewidth=0.5)
    plt.plot(S[I], initial_value(x)[I], 'k-', label='Payoff', linewidth=0.5)
    plt.xlabel('s')
    plt.ylabel('Option price')
    plt.legend(loc='upper right')
    plt.savefig('price.eps', format='eps')
    plt.show()
