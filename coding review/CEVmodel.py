import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def assembleMatrixA(N, alpha, R):
    h  = R/(N+1)
    x  = np.array([i*h for i in range(N+1)])
    w1 = (1-1/np.sqrt(3))/2
    w2 = (1+1/np.sqrt(3))/2
    p1 = w1*h
    p2 = w2*h
    diag        = np.zeros(N+1)
    diag[1:]    = 1/(2*h)*(alpha(x[:-1]+p1)+alpha(x[:-1]+p2)+alpha(x[1:]+p1)+alpha(x[1:]+p2))
    diag[0]     = 1/(2*h)*(alpha(p1)+alpha(p2))
    diag_down   = -1/(2*h)*(alpha(x[:-1]+p1)+alpha(x[:-1]+p2))
    diag_up     = diag_down
    
    A = sp.diags([diag_down,diag,diag_up], [-1,0,1])

    return A


def assembleMatrixB(N, beta, R):
    h  = R/(N+1)
    x  = np.array([i*h for i in range(N+1)])
    w1 = (1-1/np.sqrt(3))/2
    w2 = (1+1/np.sqrt(3))/2
    p1 = w1*h
    p2 = w2*h
    diag        = np.zeros(N+1)
    diag[1:]    = 1/2*(beta(x[:-1]+p1)*w1+beta(x[:-1]+p2)*w2
                       -beta(x[1:]+p1)*w2-beta(x[1:]+p2)*w1)
    diag[0]     = -1/2*(beta(p1)*w2+beta(p2)*w1)
    diag_up     = 1/2*(beta(x[:-1]+p1)*w2+beta(x[:-1]+p2)*w1)
    diag_down   = -1/2*(beta(x[:-1]+p1)*w1+beta(x[:-1]+p2)*w2)

    B = sp.diags([diag_down,diag,diag_up], [-1,0,1])
    return B


def assembleMatrixC(N, gamma, R):
    h  = R/(N+1)
    x  = np.array([i*h for i in range(N+1)])
    w1 = (1-1/np.sqrt(3))/2
    w2 = (1+1/np.sqrt(3))/2
    p1 = w1*h
    p2 = w2*h
    diag        = np.zeros(N+1)
    diag[0] = h / 2 * (gamma(p1) * w2**2 + gamma(p2) * w1**2)
    diag[1:] = h / 2 * (gamma(x[:-1] + p1) * w1**2 + gamma(x[:-1] + p2) * w2**2)
    diag[1:] += h / 2 * (gamma(x[1:] + p1) * w2**2 + gamma(x[1:] + p2) * w1**2)
    updiag = h / 2 * w1 * w2 * (gamma(x[:-1] + p1) + gamma(x[:-1] + p2))
    lodiag = updiag

    C = sp.diags([lodiag, diag, updiag], [-1, 0, 1])
    return C


def assembleMatrix(N, alpha, beta, gamma, R):
    A = assembleMatrixA(N, alpha, R)
    B = assembleMatrixB(N, beta, R)
    C = assembleMatrixC(N, gamma, R)
    return A + B + C

def buildMassCEV(N, mu, R):
    def alpha(s):
        return 0*s
    def beta(s):
        return 0*s
    def gamma(s):
        return s**(2*mu)
    
    return assembleMatrix(N, alpha, beta, gamma, R)


def buildACEV(N, sigma, rho, mu, r, R):
    def alpha(s):
        return sigma**2/2*s**(2*rho+2*mu)
    def beta(s):
        return (rho+mu)*sigma**2*s**(2*rho+2*mu-1)-r*s**(1+2*mu)
    def gamma(s):
        return r*s**(2*mu)
    
    return assembleMatrix(N, alpha, beta, gamma, R)


def initial_value(K, s):
    val = K - s
    val[val < 0] = 0
    return val


def FEM_theta(N, sigma, rho, mu, r, R, M, T, K, theta):
    k = T / M
    grid = np.linspace(0, R, N + 2)[:-1]  # exclude last point as u(R) = 0
    u_sol = initial_value(K, grid)
    MatrixM = buildMassCEV(N, mu, R)
    MatrixA = buildACEV(N, sigma, rho, mu, r, R)
    B_theta = (MatrixM + k * theta * MatrixA).tocsr()
    C_theta = MatrixM - k * (1 - theta) * MatrixA
    for _ in range(M):
        u_sol = spsolve(B_theta, C_theta @ u_sol)
    return u_sol


def test():
    r = 0.05
    K = 1
    T = 1
    R = 4
    sigma = 0.3
    theta = 0.5
    N = 999
    rho = 0.5
    mu = -0.25
    M = 1000
    print(FEM_theta(N, sigma, rho, mu, r, R, M, T, K, theta)[250])


if __name__ == "__main__":
    test()
