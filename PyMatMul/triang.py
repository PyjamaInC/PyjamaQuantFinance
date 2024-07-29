import numpy as np

def testGen(n, val):
    A = np.arange(val, val + n*n).reshape(n, n)
    A = np.sqrt(A)
    bs = (A[0, :])**2.1
    return A, bs

def forwardSub(L, bs):
    n = bs.size
    xs = np.zeros(n)
    for i in range(n):
        xs[i] = (bs[i] - L[i, :i]@xs[:i])/L[i, i]
    return xs

def backSub(U, bs):
    n = bs.size
    xs = np.zeros(n)
    for i in reversed(range(n)):
        xs[i] = (bs[i] - U[i, i+1:]@xs[i+1 :])/U[i, i]
    return xs

def testSolve(f, A, bs):
    xs = f(A, bs); print(xs)
    xs = np.linalg.solve(A, bs); print(xs)
