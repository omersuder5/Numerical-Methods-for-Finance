import numpy as np
import numpy.linalg as lin

# Implementation of the PSOR algorithm


def criterion(A, b, c, x, tol):
    u1 = x - c
    u2 = A @ x - b

    critOrtho = np.abs(np.dot(u1, u2)) < tol
    crit1 = np.all(u1 > -tol)
    crit2 = np.all(u2 > -tol)
    return critOrtho and crit1 and crit2



def PSOR(A, b, c, x0):
    # Algorithm Parameters
    omega = 1.5
    epsilon = 1e-6
    maxit = 1000
    x=x0
    it=0

    L = np.tril(A,k=-1)
    U = np.triu(A,1)
    D = np.diag(A)
    
    while (it<maxit) and not(criterion(A,b,c,x,epsilon)):
        it+=1
        x_plus1 = x.copy()
        for i in range(len(x0)):
            x_plus1[i] = 1/D[i]*(b[i]-np.dot(L[i,:],x_plus1)-np.dot(U[i,:],x))
            x_plus1[i] = np.maximum(c[i],x[i]+omega*(x_plus1[i]-x[i]))
        x=x_plus1.copy()
    if not criterion(A,b,c,x,epsilon):
        print("Warning: PSOR did not converge")
    return x


# Test the algorithm


def testPSOR():
    # Size of the test:
    N = 15
    a1 = np.ones(N)
    a2 = -np.ones(N - 1) * 1 / 3
    A = np.diag(a1) + np.diag(a2, k=-1) + np.diag(a2, k=1)
    # This is a diagonally dominant, positive definite matrix, so the algorithm
    # must converge in theory.
    u = np.random.rand(N)
    # Prepare test data
    x = u.copy()
    x[u < 0.5] = 0
    b = A @ x
    b[u < 0.5] -= u[u < 0.5]
    c = np.zeros(N)
    x0 = c
    xguess = PSOR(A, b, c, x0)
    print("L-infinity error:", lin.norm(x - xguess, ord=np.inf))
    # In theory, x and xguess should be very close.


if __name__ == "__main__":
    testPSOR()
