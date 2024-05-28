######################################################################
######################################################################
#
# Template file
# AmericanCEV.py:
#
# Pricing of an American option in a CEV model
#
#
######################################################################
######################################################################


######################################################################
# Importing modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from PDAS import PDAS  # Make sure PDAS.py is in the same folder as this file.
# Make sure assembleMatrix.py is in the same folder as this file.
from assembleMatrix import assembleMatrix

######################################################################
# Spot-price grid

def hval(N, R):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the spot-price

    Returns the log-price grid step"""
    return R/(N+1)

    #
    # Your code here
    #
    # return ...

def sgrid(N, R):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the spot-price

    Returns the spot-price grid [s0,s1,...,sN,sN+1]"""
    return np.linspace(-R, R, N+2)

    #
    # Your code here
    #
    # return ...


######################################################################
# Time grid


def Mval(N, tau, R):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the spot price
    tau: size of the time interval for the theta-scheme

    Returns the number M of time steps to perform in the theta-scheme """

    h = hval(N, R)
    return np.ceil[tau/h]
    #
    # Your code here
    #
    # return ...


def kval(tau,M):
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
# FEM assembling routines


def assemble_M(N, R, mu):
    """Inputs:
    N: dimension of VN
    R: spot price truncation parameter
    mu: parameter of the CEV model

    Returns the NxN matrix M^{CEV}
    """
    A=0
    B=0
    s=sgrid(N, R)[1:-1]
    C=s**(2*mu)
    return assembleMatrix(N, 0, R, A, B, C)
    #
    # Your code here
    #
    # return ...


def assemble_A(N, R, r, sigma, rho, mu):
    """Inputs:
    N: dimension of VN
    R: spot price truncation parameter
    r: riskless interest rate
    sigma, rho: CEV volatility = sigma*rho^{s}
    mu: parameter of the CEV model

    Returns the NxN matrix A^{CEV}
    """
    s = sgrid(N, R)[1:-1]
    A=1/2*sigma**2*s**(2*(rho+mu))
    B=sigma**2*(rho+mu)*s**(2*(rho+mu)-1)-r*s**(1+2*mu)
    C = r*s**(2*mu)
    return assembleMatrix(N, 0, R, A, B, C)
    #
    # Your code here
    #
    # return ...


# Matrix B
def assemble_B(M, A, k):
    """Inputs:
    M: Matrix M^{CEV} (mass matrix)
    A: matrix A^{CEV}
    k: time-step size

    Returns the matrix B defined in the exam
    """

    return M + k*A
    #
    # Your code here
    #
    # return ...

######################################################################
# FEM plotting routine


def plotFE(R, vec, lab, leftLim, rightLim):
    """Inputs:
    R: spot price truncation constant
    vec: a 1-dimensional numpy array
    lab: a chain of character (used in legend of plot)
    leftLim, rightLim: interval on which the function is plotted

    Plots the graph of the FE function whose coordinates are given by vec.
    """
    N = vec.size
    si = sgrid(N, R)
    vals = 0*si
    vals[1:-1] = vec
    ind = (si >= leftLim) & (si <= rightLim)
    plt.plot(si[ind], vals[ind], label=lab)


######################################################################
# Option pricing routines

# American option
def computeCEVAmericanOption(N, R, r, sigma, rho, mu, T, gvec):
    """ Inputs:
    N: number of dofs
    R: such that G = (0,R)
    r: riskless interest rate
    rho, sigma: sigma rho^s is the underlying volatility
    mu: parameter of the FEM
    T: option maturity
    gvec: vector (g(s_1),...,g(s_N)), where g is the payoff and
    s_i are the spot price grid points.

    Output:
    Vector v approx (v(T,s_1),...,v(T,s_N)) where
    v(t,s) is the value of the American option at time-to-maturity t and spot price s
    """
    s = sgrid(N, R)[1:-1]
    M = Mval(N, T, R)
    k = kval(T, M)
    A_mat = assemble_A(N, R, r, sigma, rho, mu)
    M_mat = assemble_M(N, R, mu)
    B = assemble_B(M_mat, A_mat, k)
    G=np.dot(A_mat,gvec(s))
    u_sol = gvec(s)
    for m in range(M):
        Fm = k*G + np.dot(M_mat,u_sol)
        u_sol=PDAS(B, Fm, u_sol, 1e-10)
    return u_sol

    #
    # Your code here
    #
    # return ...



print('Pricing of an American put option')

# Grid definition
R = 10  # Truncation parameter in spot price
N = 1000  # Number of dofs
si = sgrid(N,R) # spot-price grid [s0,...,sN+1]
sdof = si[1:-1] # array [s1,...,sN]

print('Resolution on (0,{R}) with {N} degrees of freedom'.format(R=R, N=N))

# Market model: CEV
rho = 0.7  # CEV model constant rho
sigma = 0.3  # so that sigma(s) = sigma*rho^{s}
mu = -0.4 # parameter mu of the method
r = 0.05 # Interest rate

print('CEV model: rho = {rho}, sigma = {sigma}, r = {r}'.format(
    rho=rho, sigma=sigma, r=r))
print(
    'Setting parameter mu to {mu} -> rho + mu = {rhopmu}'.format(mu=mu, rhopmu=rho+mu))

# Option definition
T = 1  # Maturity
K = 0  # Strike price
# Payoff
def gput(S):
    K = 0 # Defined two times to avoid constant variables overwriting
    return np.maximum(K - S, 0)
    #
    # Your code here
    #
    # return ...


print('Put option with maturity {T} and strike price {K}'.format(T=T, K=K))
gvec = gput(sdof) # numpy array [g(s1),...,g(sN)] where g is the payoff

# Area of interest
leftLim = 0
rightLim = R
print('Pricing for stock prices s in ({l},{r})'.format(l=leftLim, r=rightLim))

# Computation of option values:
print('Computing put option price...')
Vam = 0 # Nx1 numpy array approximating [Vam(t,s1),...,Vam(t,sN)]
print('... Done!')


# Plot American put
print('Plotting results')
plotFE(R, Vam, 'CEV American put',leftLim, rightLim)
ind = (sdof > leftLim)*(sdof < rightLim)
plt.plot(sdof[ind], gput(sdof[ind]), 'k--', label='Payoff')
plt.xlabel('Spot price s')
plt.ylabel('Value')
plt.legend()
plt.show()
