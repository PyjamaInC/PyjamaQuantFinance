import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from mpl_toolkits import mplot3d

def generate_gbm_paths(no_of_paths, no_of_steps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [no_of_paths, no_of_steps])
    X = np.zeros([no_of_paths, no_of_steps + 1])
    time = np.zeros([no_of_steps + 1])

    X[:, 0] = np.log(S_0)
    dt = T / float(no_of_steps)
    
    for i in range(no_of_steps):
        if no_of_paths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        X[:, i+1] = X[:, i] + (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z[:, i]
        time[i+1] = time[i] + dt
    
    S = np.exp(X)
    return {"time": time, "X": X, "S": S}


def Martingale(Paths, Steps, Time, S0, seed):
    np.random.seed(seed)
    NoOfPaths = Paths
    NoOfSteps = Steps
    T = Time
    r = 0.05
    mu = 0.05
    sigma = 0.1
    S_0 = S0

    M = lambda t: np.exp(r*t)

    pathsQ = generate_gbm_paths(NoOfPaths, NoOfSteps, T, r, sigma, S_0)
    S_Q = pathsQ["S"]
    pathsP = generate_gbm_paths(NoOfPaths, NoOfSteps, T, mu, sigma, S_0)
    S_P = pathsP["S"]
    time = pathsQ["time"]

    discountS_Q = np.zeros([NoOfPaths, NoOfSteps+1])
    discountS_P = np.zeros([NoOfPaths, NoOfSteps+1])
    for i, ti in enumerate(time):
        discountS_P[:, i] = S_P[:, i]/M(ti)
        discountS_Q[:, i] = S_Q[:, i]/M(ti)

    plt.figure(1)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    
    # we calculate the expected output for a process to be a martingale
    ES_Q = lambda t: S_0 * np.exp(r*t)/M(t)
    plt.plot(time, ES_Q(time), 'r--')
    plt.plot(time, np.transpose(discountS_Q), 'blue')
    plt.legend(['E^Q[S(t)/M(t)]', 'paths S(t)/M(t)'])

    plt.figure(2)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(time, ES_Q(time), 'r--')
    plt.plot(time, np.transpose(discountS_P), 'blue')
    plt.legend(['E^P[S(t)/M(t)]', 'paths S(t)/M(t)'])

    #return discountS_P, discountS_Q