######################################################################
######################################################################
#
# Template file
# CompoundBS.py:
#
# Pricing of a compound option in a Black-Scholes model
#
#
######################################################################
######################################################################


######################################################################
# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

######################################################################
# Black-Scholes market: definitions and explicit solutions


def bs_formula_C(r, sigma, s, t, T, K):
    """Inputs:
    r: riskless interest rate
    sigma: underlying volatility
    s: spot price
    t: time at which the call option is evaluated
    T: call maturity
    K: strike price

    Returns the value V(t,s) of the European call with the above parameters"""

    tau = T - t
    if tau == 0:
        tau = 1e-10  # Replace by very small value to avoid runtime warning
    d1 = (np.log(s/K) + (0.5*sigma**2+r)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    V = s*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    return V

######################################################################
# log-price (uniform) grid


def hval(N, R):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the log-price

    Returns the log-price grid step"""
    return 2*R/(N+1)
    #
    # Your code here
    #
    # return ...


def xgrid(N, R):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the log-price

    Returns the numpy array containing the log-price uniform grid
    [x0, x1, ..., xN, xN+1]"""
    return np.array([-R+i*hval(N,R) for i in range(N+2)])
    #
    # Your code here
    #
    # return ...

######################################################################
# time grid


def Mval(N, R, tau):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the log-price
    tau: size of the time interval for the theta-scheme

    Returns the number M of time steps to perform in the theta-scheme """

    h = hval(N, R)
    return np.ceil(tau/h)
    #
    # Your code here
    #
    # return ...


def kval(tau, M):
    """Inputs:
    tau: size of the time interval for the theta-scheme
    M: number of steps to perform in the theta-scheme


    Returns the time-step size k"""
    return tau/M

    #
    # Your code here
    #
    # return ...

######################################################################
# FEM assembling routines:


def tridiag(N, a, b, c):
    """Inputs:
    N: a natural number
    a,b,c: real numbers

    Returns the NxN sparse tridiagonal matrix
    [b c 0 0 0 ... 0]
    [a b c 0 0 ... 0]
    [0 a b c 0 ... 0]
    ...
    [0 0 0 ... a b c]
    [0 0 0 ...   a b]
    """

    diaga = a*np.ones(N-1)
    diagb = b*np.ones(N)
    diagc = c*np.ones(N-1)
    T = sp.diags([diaga, diagb, diagc], [-1, 0, 1])
    return T


def assemble_M(N):
    """Inputs:
    N: dimension of VN

    Returns the NxN matrix M
    """
    h= 2*R/(N+1)
    a=2/3*h
    b=1/6*h
    return tridiag(N,b,a,b)
    #
    # Your code here
    #
    # return ...


def assemble_S(N):
    """Inputs:
    N: dimension of VN

    Returns the NxN matrix S
    """
    h= 2*R/(N+1)
    a=2/h
    b=-1/h
    return tridiag(N,b,a,b)

    #
    # Your code here
    #
    # return ...


def assemble_W(N):
    """Inputs:
    N: dimension of VN

    Returns the NxN matrix W
    """
    a=0
    b=-1/2
    c=1/2
    return tridiag(N,c,a,b)
    #
    # Your code here
    #
    # return ...


def assemble_A(N, r, sigma):
    """Inputs:
    N: dimension of VN
    r: riskless interest rate
    sigma: underlying volatility

    Returns the NxN matrix A^{BS}
    """
    S = assemble_S(N)
    W = assemble_W(N)
    M = assemble_M(N)
    return 1/2*sigma**2*S+(sigma**2/2-r)*W+r*M

    #
    # Your code here
    #
    # return ...

######################################################################
# FEM plotting routine:


def plotFE(R, vec, lab, leftLim, rightLim):
    """Inputs:
    R: truncation parameter for log-price
    vec: a 1-dimensional numpy array
    lab: a chain of character (used in legend of plot)
    leftLim, rightLim: interval on which the function is plotted

    Plots the graph of the FE function whose coordinates are given by vec.
    """

    N = vec.size
    xi = xgrid(N, R)
    vals = 0*xi
    vals[1:-1] = vec
    ind = (np.exp(xi) > leftLim) & (np.exp(xi) < rightLim)
    plt.plot(np.exp(xi[ind]), vals[ind], label=lab)

######################################################################
# Theta-scheme:


def assemble_B(M, A, theta, k):
    """Inputs:
    M: Matrix M (mass matrix)
    A: matrix A^{BS}
    theta: parameter of the theta-scheme
    k: time-step size

    Returns the matrix B_{theta}
    """
    return M+k*theta*A

    #
    # Your code here
    #
    # return ...


def assemble_C(M, A, theta, k):
    """Inputs:
    M: Matrix M (mass matrix)
    A: matrix A^{BS}
    theta: parameter of the theta-scheme
    k: time-step size

    Returns the matrix C_{theta}
    """
    return M-(1-theta)*k*A

    #
    # Your code here
    #
    # return ...


######################################################################
# European option pricing routine
def computeBSEuropeanOption(N, R, r, sigma, t, T, gvec, theta):
    """ Inputs:
    N: dimension of VN
    R: truncation parameter for the log-price
    r: riskless interest rate
    sigma: underlying volatility
    t: time at which the option is priced
    T: option maturity
    gvec: vector (g(x1),...,g(xN)), where g is the payoff and xi are the grid points
    theta: parameter of the theta-scheme

    Returns a Nx1 numpy array approximating [V(t,s_1),...,V(t,s_N)] where
    V(t,s) is the value of a European option at time t and spot price s with
    payoff g, maturity T under a BS model with parameters sigma and r.
    """
    h = hval(N,R)
    x = xgrid(N,R)[1:-1]
    u_sol = gvec(x)
    M = Mval(N,R,T-t)

    B = assemble_B(assemble_M(N), assemble_A(N, r, sigma), theta, kval(T-t, Mval(N, R, T-t)))
    C = assemble_C(assemble_M(N), assemble_A(N, r, sigma), theta, kval(T-t, Mval(N, R, T-t)))

    for m in range(M):
        u_sol = spsolve(B,np.dot(C,u_sol))
    return u_sol
    #
    # Your code here
    #
    # return ...


######################################################################
# Computation of the European option V1:
print('Underlying European call pricing')

# Log-price grid definition
N = 1000 # dimension of VN
R = 5.5  # Truncation constant for log price
xi = xgrid(N,R) # log-price grid [x0,...,xN+1]
si = 0 # spot-price grid
xdof = xi[1:-1] # degrees of freedom [x1,...,xN]
sdof = 0 # array [s1,...,sN]

print(f'Resolution for stock prices in ({np.exp(-R)},{np.exp(R)}) with {N} degrees of freedom')

# Market model: B-S
r = 0.01# riskless interest rate
sigma = 0.3 # asset volatility

print(f'Black-Scholes model with r = {r} and sigma = {sigma}')

# Call option parameters:
t = 0  # Pricing option at time t
T1 = 1  # Maturity
K1 = 0  # Strike price
def g1(S):
    K1 = 0 # Defined two times to avoid constant variables overwriting
    return np.maximum(S-K1, 0  )
    #
    # Your code here
    #
    # return ...

g1vec = 0 # vector of values [g1(s1),...,g1(sN)]

print(f'Underlying call option with maturity {T1} and strike price {K1}')

# Computation of underlying call option value:
print('Pricing call option...')
theta = 0.5  # Value of theta in the theta scheme
V1 = 0 # Nx1 numpy array approximating [V1(t,s1),...,V1(t,sN)]
V1exact = bs_formula_C(r,sigma,si,t,T1,K1) # (N+2)x1 Numpy array with exact values [V1(t,s0),V1(t,s1)...,V1(t,sN),V1(t,sN+1)]
print('... Done!')

######################################################################
# Plotting of underlying option

# Area of interest in stock price
leftLim = 0
rightLim = 0
print(f'Plotting underlying option for stock prices s in ({leftLim},{rightLim})...')
ind = (si > leftLim) & (si < rightLim)
plotFE(R, V1, 'FEM approximation', leftLim, rightLim)
plt.plot(si[ind], V1exact[ind], 'r--', label='B-S formula')
plt.plot(si[ind], g1(si[ind]), 'k--', label='payoff')
plt.legend()
plt.xlabel('Spot price')
plt.ylabel('Option value')
plt.title('Underlying European call option pricing')
plt.show()
print('...Done!\n')

######################################################################
# Convergence graph

print('convergence graph')
def orderOfConvergence(R, r, sigma, T, g, theta, Vexact, leftLim, rightLim):
    """ Inputs:
    R: truncation parameter for log-prices
    r: riskless interest rate
    sigma: underlying volatility
    T: option maturity
    g: Python handle representing payoff function
    Vexact: Python handle which returns the exact value of the option.
    Vexact must accept numpy array representing a list of spot prices.
    leftLim, rightLim: interval in which the error is calulated

    Plots a convergence graph for the Linfty error.
    """

    imin = 7
    imax = 12
    N = 2**np.arange(imin, imax)
    h = 2*R/(N+1)

    err = np.zeros(imax-imin)
    for i in range(imin, imax):
        N = 2**i
        xi = xgrid(N, R)
        xdof = xi[1:-1]
        sdof = np.exp(xdof)
        gvec = g(sdof)
        t = 0
        V = computeBSEuropeanOption(N, R, r, sigma, t, T, gvec, theta)
        ind = (sdof > leftLim) & (sdof < rightLim)
        V = V[ind]
        Vex = Vexact(sdof[ind])
        err[i-imin] = np.max(np.abs(Vex - V))
    plt.loglog(h, err, label='Linfty error')
    plt.loglog(h, h, 'k--', label='O(h)')
    plt.loglog(h, h**2, 'g--', label='O(h^2)')
    plt.legend()
    plt.xlabel('h')
    plt.ylabel('error')
    plt.title(f'Convergence graph with theta = {theta}')
    plt.show()
    return err

# Exact solution handle
def VexactCall(S):
    return bs_formula_C(r, sigma, S, 0, T1, K1)

theta = 0 # This value can be modified
print(f'Plotting convergence graph for theta = {theta}...')
orderOfConvergence(R, r, sigma, T1, g1, theta, VexactCall, leftLim, rightLim)
print('...Done!\n')


######################################################################
# Computation of the Compound option Vc:
print('Pricing compound option')

# Definition of the compound option
t = 0  # Pricing compound at t
Tc = 0  # Compound maturity
Kc = 0  # Strike price for compound
def gc(S):
    Kc = 0
    return np.maximum(S-Kc, 0)

print(f'Compound call option with maturity Tc = {Tc} and strike price Kc = {Kc}')

# Computation of the compound option value
print('Pricing underlying call at time Tc...')
theta = 0 # parameter of the theta-scheme for the computation of V1 and Vc
V1 = 0 # Nx1 numpy array approximating [V1(Tc,s1),...,V1(Tc,sN)]
print('...Done!')
print('Pricing compound call...')
Gvec = 0 # Nx1 numpy array [gc(V1(Tc,s1)),...,gc(V1(Tc,sN))]
Vc = 0 # Nx1 numpy array approximating [Vc(0,s1),...,Vc(0,sN)]
print('...Done!')

######################################################################
# Plotting of compound option vs underlying call.

leftLim = 0
rightLim = 0
print(f'Plotting compound vs underlying call for stock prices s in ({leftLim},{rightLim})')

plotFE(R, Vc, 'Compound option value', leftLim, rightLim)
plotFE(R, V1, 'Underlying call', leftLim, rightLim)
plotFE(R, Gvec, 'Compound payoff', leftLim, rightLim)
plt.legend()
plt.xlabel('Spot price')
plt.ylabel('Option value')
plt.title('Compound call option pricing')
plt.show()
print('Done')
