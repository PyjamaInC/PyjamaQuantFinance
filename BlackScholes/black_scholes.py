import numpy as np
import matplotlib.pyplot as plt
import enum
import scipy.stats as st
from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator
import GBMSimulation as gbms

class OptionType(enum.Enum):
    CALL_OPT = 1.0
    PUT_OPT = -1.0

def Black_Scholes(Option_Type, S_0, K, sigma, t, T, r):

    # additional revision
    epsilon = 1e-8
    if T - t < epsilon:
        if Option_Type == OptionType.CALL_OPT:
            return np.maximum(S_0 - K, 0)
        elif Option_Type == OptionType.PUT_OPT:
            return np.maximum(K - S_0, 0)
        
    # depend on the range of K, we will create a column vector (array) to hold all elements in K, each one being a row
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5*np.power(sigma, 2.0)) * (T-t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T-t)
    if Option_Type == OptionType.CALL_OPT:
        bs_val = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T - t))
    if Option_Type == OptionType.PUT_OPT:
        bs_val = st.norm.cdf(-d2) * K * np.exp(-r * (T - t)) - st.norm.cdf(-d1) * S_0

    return bs_val

def Delta_BS(Option_Type, S_0, K, sigma, t, T, r):

    # if t - T > 10e-20 and T-t < 10e-7:
    #     t = T

    # additional revision
    epsilon = 1e-8
    if T - t < epsilon:
        if Option_Type == OptionType.CALL_OPT:
            return np.where(S_0 > K, 1.0, 0.0)
        elif Option_Type == OptionType.PUT_OPT:
            return np.where(S_0 < K, -1.0, 0.0)
        
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5*np.power(sigma, 2.0)) * (T-t)) / (sigma * np.sqrt(T - t))
    if Option_Type == OptionType.CALL_OPT:
        val = st.norm.cdf(d1)
    if Option_Type == OptionType.PUT_OPT:
        val = st.norm.cdf(d1) - 1.0

    return val

def Gamma_BS(S_0, K, sigma, t, T, r):

    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5*np.power(sigma, 2.0)) * (T-t)) / (sigma * np.sqrt(T - t))
    return st.norm.pdf(d1) / (S_0 * sigma * np.sqrt(T - t))

def Vega_BS(S_0, K, sigma, t, T, r):
    d1 = (np.log(S_0 / K) + (r + 0.5*np.power(sigma, 2.0)) * (T-t)) / (sigma * np.sqrt(T - t))
    return S_0 * st.norm.pdf(d1) * np.sqrt(T-t)


def simulate(
        NoOfPaths,
        NoOfSteps,
        T,
        r,
        sigma,
        S_0,
        K,
        Option_Type = OptionType.CALL_OPT
):
    np.random.seed(42)
    Paths = gbms.genGBMPaths(NoOfPaths, NoOfSteps, T, r, sigma, S_0)
    time = Paths["time"]
    S = Paths["S"]

    C = lambda time, Strike, Curr: Black_Scholes(Option_Type, Curr, Strike, sigma, time, T, r)
    Delta = lambda time, Strike, Curr: Delta_BS(Option_Type, Curr, Strike, sigma, time, T, r)
    
    # a matrix of Paths rows and Steps (t) columns
    PnL = np.zeros([NoOfPaths, NoOfSteps + 1])
    initDelta = Delta(0.0, K, S_0)
    PnL[:, 0] = C(0.0, K, S_0) - initDelta * S_0

    CallM = np.zeros([NoOfPaths, NoOfSteps+1])
    CallM[:, 0] = C(0.0, K, S_0)
    DeltaM = np.zeros([NoOfPaths, NoOfSteps+1])
    DeltaM[:,0] = Delta(0.0, K, S_0)

    for i in range(1, NoOfSteps + 1):
        dt = time[i] - time[i-1]
        delta_prev = Delta(time[i-1], K, S[:, i-1])
        delta_curr = Delta(time[i], K, S[:, i])

        PnL[:, i] = PnL[:, i-1]*np.exp(r*dt) - (delta_curr - delta_prev)*S[:, i]
        CallM[:, i] = C(time[i], K, S[:, i])
        DeltaM[:, i] = delta_curr

    PnL[:, -1] = PnL[:, -1] - np.maximum(S[:, -1]-K, 0) + DeltaM[:, -1]*S[:, -1]

    path_id = 13
    plt.figure(1)
    plt.figure(1)
    plt.plot(time,S[path_id,:])
    plt.plot(time,CallM[path_id,:])
    plt.plot(time,DeltaM[path_id,:])
    plt.plot(time,PnL[path_id,:])
    plt.legend(['Stock','CallPrice','Delta','PnL'])
    plt.grid()

    plt.figure(2)
    plt.hist(PnL[:,-1],50)
    plt.grid()
    plt.xlim([-0.1,0.1])
    plt.title('histogram of P&L')

    for i in range(0,NoOfPaths):
        print('path_id = {0:2d}, PnL(t_0)={1:0.4f}, PnL(Tm-1) ={2:0.4f},S(t_m) = {3:0.4f}, max(S(tm)-K,0)= {4:0.4f}, PnL(t_m) = {5:0.4f}'.format(i,PnL[0,0],
              PnL[i,-2],S[i,-1],np.max(S[i,-1]-K,0),PnL[i,-1]))
        
def plot_CallOpt(
        NoOfPaths,
        NoOfSteps,
        T,
        r,
        sigma,
        S_0,
        K,
        seed = 3,
        pathID = 12,
):
    # plotting a vega suface values of a call with specified parameters
    np.random.seed(seed)
    Paths = gbms.genGBMPaths(NoOfPaths, NoOfSteps, T, r, sigma, S_0)
    time = Paths["time"]
    S = Paths["S"]

    # 1D array of 50 elements and 100 elements
    S0_grid = np.linspace(S_0/100.0, 1.5 * S_0, 50)
    timeGrid = np.linspace(0.02, T - 0.02, 100)

    C = lambda time, Curr: Black_Scholes(OptionType.CALL_OPT, Curr, K, sigma, time, T, r)
    P = lambda time, Curr: Black_Scholes(OptionType.PUT_OPT, Curr, K, sigma, time, T, r)
    # Delta = lambda time, Curr: Delta_BS(Option_Type, Curr, K, sigma, time, T, r)
    # Gamma = lambda time, Curr: Gamma_BS(Curr, K, sigma, time, T, r)
    # Vega = lambda time, Curr: Vega_BS(Curr, K, sigma, time, T, r)

    # rows: timeGrid - 100, columns: S0_grid: 50
    CallOpt_mat = np.zeros([len(timeGrid), len(S0_grid)])
    # PutOpt_mat = np.zeros([len(timeGrid), len(S0_grid)])
    # Delta_mat = np.zeros([len(timeGrid), len(S0_grid)])
    # Gamma_mat = np.zeros([len(timeGrid), len(S0_grid)])
    # Vega_mat = np.zeros([len(timeGrid), len(S0_grid)])
    T_mat = np.zeros([len(timeGrid), len(S0_grid)])
    S0_mat = np.zeros([len(timeGrid), len(S0_grid)])

    for i in range(0, len(timeGrid)):
        T_mat[i, :] = timeGrid[i]
        S0_mat[i, :] = S0_grid
        CallOpt_mat[i, :] = C(timeGrid[i], S0_grid)
        # PutOpt_mat[i, :] = P(timeGrid[i], S0_grid)
        # Delta_mat[i, :] = Delta(timeGrid[i], S0_grid)
        # Gamma_mat[i, :] = Gamma(timeGrid[i], S0_grid)
        # Vega_mat[i, :] = Vega(timeGrid[i], S0_grid)

    plt.figure(1)
    plt.plot(time, np.squeeze(S[pathID, :]))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(T, K, "ok")

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')  # Changed this line
    ax.plot_surface(T_mat, S0_mat, CallOpt_mat, color=[1, 0.5, 1])
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.title('Call option surface')
    Finterp = RegularGridInterpolator((timeGrid[0:], S0_grid), CallOpt_mat)
    v = np.zeros([len(time), 1])
    vTemp = []
    timeTemp = []
    pathTemp = []
    for j in range(5, len(time)):
        if time[j] > timeGrid[0] and time[j] < timeGrid[-1]:
            v[j] = Finterp([time[j], S[pathID, j]])
            vTemp.append(Finterp([time[j], S[pathID, j]])[0])
            timeTemp.append(time[j])
            pathTemp.append(S[pathID, j])
    
    ax.plot3D(np.array(timeTemp), np.array(pathTemp), np.array(vTemp), 'blue')
        
def plot_PutOpt(
        NoOfPaths,
        NoOfSteps,
        T,
        r,
        sigma,
        S_0,
        K,
        seed = 3,
        pathID = 12,
):
    # plotting a vega suface values of a call with specified parameters
    np.random.seed(seed)
    Paths = gbms.genGBMPaths(NoOfPaths, NoOfSteps, T, r, sigma, S_0)
    time = Paths["time"]
    S = Paths["S"]

    # 1D array of 50 elements and 100 elements
    S0_grid = np.linspace(S_0/100.0, 1.5 * S_0, 50)
    timeGrid = np.linspace(0.02, T - 0.02, 100)

    P = lambda time, Curr: Black_Scholes(OptionType.PUT_OPT, Curr, K, sigma, time, T, r)
    PutOpt_mat = np.zeros([len(timeGrid), len(S0_grid)])
    T_mat = np.zeros([len(timeGrid), len(S0_grid)])
    S0_mat = np.zeros([len(timeGrid), len(S0_grid)])

    for i in range(0, len(timeGrid)):
        T_mat[i, :] = timeGrid[i]
        S0_mat[i, :] = S0_grid
        PutOpt_mat[i, :] = P(timeGrid[i], S0_grid)

    plt.figure(1)
    plt.plot(time, np.squeeze(S[pathID, :]))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(T, K, "ok")

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')  # Changed this line
    ax.plot_surface(T_mat, S0_mat, PutOpt_mat, color=[1, 0.5, 1])
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.title('Put option surface')
    Finterp = RegularGridInterpolator((timeGrid[0:], S0_grid), PutOpt_mat)
    v = np.zeros([len(time), 1])
    vTemp = []
    timeTemp = []
    pathTemp = []
    for j in range(5, len(time)):
        if time[j] > timeGrid[0] and time[j] < timeGrid[-1]:
            v[j] = Finterp([time[j], S[pathID, j]])
            vTemp.append(Finterp([time[j], S[pathID, j]])[0])
            timeTemp.append(time[j])
            pathTemp.append(S[pathID, j])
    
    ax.plot3D(np.array(timeTemp), np.array(pathTemp), np.array(vTemp), 'blue')

        
def plot_Delta(
        NoOfPaths,
        NoOfSteps,
        T,
        r,
        sigma,
        S_0,
        K,
        Option_Type = OptionType.CALL_OPT,
        seed = 3,
        pathID = 12,
):
    # plotting a vega suface values of a call with specified parameters
    np.random.seed(seed)
    Paths = gbms.genGBMPaths(NoOfPaths, NoOfSteps, T, r, sigma, S_0)
    time = Paths["time"]
    S = Paths["S"]

    # 1D array of 50 elements and 100 elements
    S0_grid = np.linspace(S_0/100.0, 1.5 * S_0, 50)
    timeGrid = np.linspace(0.02, T - 0.02, 100)

    Delta = lambda time, Curr: Delta_BS(Option_Type, Curr, K, sigma, time, T, r)
    
    Delta_mat = np.zeros([len(timeGrid), len(S0_grid)])
    T_mat = np.zeros([len(timeGrid), len(S0_grid)])
    S0_mat = np.zeros([len(timeGrid), len(S0_grid)])

    for i in range(0, len(timeGrid)):
        T_mat[i, :] = timeGrid[i]
        S0_mat[i, :] = S0_grid
        Delta_mat[i, :] = Delta(timeGrid[i], S0_grid)
        

    plt.figure(1)
    plt.plot(time, np.squeeze(S[pathID, :]))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(T, K, "ok")

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')  # Changed this line
    ax.plot_surface(T_mat, S0_mat, Delta_mat, color=[1, 0.5, 1])
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.title('Delta option surface')
    Finterp = RegularGridInterpolator((timeGrid[0:], S0_grid), Delta_mat)
    # v = np.zeros([len(time), 1])
    vTemp = []
    timeTemp = []
    pathTemp = []
    for j in range(5, len(time)):
        if time[j] > timeGrid[0] and time[j] < timeGrid[-1]:
            # v[j] = Finterp([time[j], S[pathID, j]])
            vTemp.append(Finterp([time[j], S[pathID, j]])[0])
            timeTemp.append(time[j])
            pathTemp.append(S[pathID, j])
    
    ax.plot3D(np.array(timeTemp), np.array(pathTemp), np.array(vTemp), 'blue')
    ax.view_init(30, 120)

        
def plot_Vega(
        NoOfPaths,
        NoOfSteps,
        T,
        r,
        sigma,
        S_0,
        K,
        Option_Type = OptionType.CALL_OPT,
        seed = 3,
        pathID = 12,
):
    # plotting a vega suface values of a call with specified parameters
    np.random.seed(seed)
    Paths = gbms.genGBMPaths(NoOfPaths, NoOfSteps, T, r, sigma, S_0)
    time = Paths["time"]
    S = Paths["S"]

    # 1D array of 50 elements and 100 elements
    S0_grid = np.linspace(S_0/100.0, 1.5 * S_0, 50)
    timeGrid = np.linspace(0.02, T - 0.02, 100)

    Vega = lambda time, Curr: Vega_BS(Curr, K, sigma, time, T, r)
    
    Vega_mat = np.zeros([len(timeGrid), len(S0_grid)])
    T_mat = np.zeros([len(timeGrid), len(S0_grid)])
    S0_mat = np.zeros([len(timeGrid), len(S0_grid)])

    for i in range(0, len(timeGrid)):
        T_mat[i, :] = timeGrid[i]
        S0_mat[i, :] = S0_grid
        Vega_mat[i, :] = Vega(timeGrid[i], S0_grid)
        

    plt.figure(1)
    plt.plot(time, np.squeeze(S[pathID, :]))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(T, K, "ok")

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')  # Changed this line
    ax.plot_surface(T_mat, S0_mat, Vega_mat, color=[1, 0.5, 1])
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.title('Delta option surface')
    Finterp = RegularGridInterpolator((timeGrid[0:], S0_grid), Vega_mat)
    # v = np.zeros([len(time), 1])
    vTemp = []
    timeTemp = []
    pathTemp = []
    for j in range(5, len(time)):
        if time[j] > timeGrid[0] and time[j] < timeGrid[-1]:
            # v[j] = Finterp([time[j], S[pathID, j]])
            vTemp.append(Finterp([time[j], S[pathID, j]])[0])
            timeTemp.append(time[j])
            pathTemp.append(S[pathID, j])
    
    ax.plot3D(np.array(timeTemp), np.array(pathTemp), np.array(vTemp), 'blue')
    ax.view_init(30, -120)

        
def plot_Gamma(
        NoOfPaths,
        NoOfSteps,
        T,
        r,
        sigma,
        S_0,
        K,
        seed = 3,
        pathID = 12,
):
    # plotting a vega suface values of a call with specified parameters
    np.random.seed(seed)
    Paths = gbms.genGBMPaths(NoOfPaths, NoOfSteps, T, r, sigma, S_0)
    time = Paths["time"]
    S = Paths["S"]

    # 1D array of 50 elements and 100 elements
    S0_grid = np.linspace(S_0/100.0, 1.5 * S_0, 50)
    timeGrid = np.linspace(0.02, T - 0.02, 100)

    Gamma = lambda time, Curr: Gamma_BS(Curr, K, sigma, time, T, r)

    # rows: timeGrid - 100, columns: S0_grid: 50
    Gamma_mat = np.zeros([len(timeGrid), len(S0_grid)])
    T_mat = np.zeros([len(timeGrid), len(S0_grid)])
    S0_mat = np.zeros([len(timeGrid), len(S0_grid)])

    for i in range(0, len(timeGrid)):
        T_mat[i, :] = timeGrid[i]
        S0_mat[i, :] = S0_grid
        Gamma_mat[i, :] = Gamma(timeGrid[i], S0_grid)
        
    plt.figure(1)
    plt.plot(time, np.squeeze(S[pathID, :]))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(T, K, "ok")

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')  # Changed this line
    ax.plot_surface(T_mat, S0_mat, Gamma_mat, color=[1, 0.5, 1])
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.title('Call option surface')
    Finterp = RegularGridInterpolator((timeGrid[0:], S0_grid), Gamma_mat)
    v = np.zeros([len(time), 1])
    vTemp = []
    timeTemp = []
    pathTemp = []
    for j in range(5, len(time)):
        if time[j] > timeGrid[0] and time[j] < timeGrid[-1]:
            v[j] = Finterp([time[j], S[pathID, j]])
            vTemp.append(Finterp([time[j], S[pathID, j]])[0])
            timeTemp.append(time[j])
            pathTemp.append(S[pathID, j])
    
    ax.plot3D(np.array(timeTemp), np.array(pathTemp), np.array(vTemp), 'blue')
    ax.view_init(30, -120)

