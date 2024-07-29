import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz

def estimate_order(y, max_p=5, max_d=2, max_q=5):
    """
    Estimate p, d, and q parameters for ARIMA model using AIC.
    """
    y = np.array(y)
    best_aic = np.inf
    best_order = (0, 0, 0)
    
    for d in range(max_d + 1):
        if d > 0:
            diff_y = np.diff(y, n=d)
        else:
            diff_y = y
        
        mean = np.mean(diff_y)
        variance = np.var(diff_y)
        n = len(diff_y)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                
                # Fit AR model
                if q == 0:
                    phi = estimate_ar_params(diff_y, p)
                    aic = compute_aic(diff_y, p, phi, mean, variance)
                # Fit MA model
                elif p == 0:
                    theta = estimate_ma_params(diff_y, q)
                    aic = compute_aic(diff_y, q, theta, mean, variance)
                # Fit ARMA model
                else:
                    phi, theta = estimate_arma_params(diff_y, p, q)
                    aic = compute_aic(diff_y, p + q, np.concatenate([phi, theta]), mean, variance)
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
    
    return best_order

def estimate_ar_params(y, p):
    """Estimate AR parameters using Yule-Walker equations."""
    r = np.correlate(y - y.mean(), y - y.mean(), mode='full')
    r = r[len(r)//2:] / len(y)
    R = toeplitz(r[:p])
    return solve_toeplitz(R, r[1:p+1])

def estimate_ma_params(y, q):
    """Estimate MA parameters using method of moments."""
    r = np.correlate(y - y.mean(), y - y.mean(), mode='full')
    r = r[len(r)//2:] / len(y)
    theta = np.zeros(q)
    for i in range(q):
        theta[i] = r[i+1] / r[0]
    return theta

def estimate_arma_params(y, p, q):
    """Estimate ARMA parameters using a simple two-step method."""
    phi = estimate_ar_params(y, p)
    resid = y - np.convolve(y, np.r_[1, -phi], mode='valid')
    theta = estimate_ma_params(resid, q)
    return phi, theta

def compute_aic(y, k, params, mean, variance):
    """Compute AIC for a model."""
    n = len(y)
    resid = y - np.convolve(y - mean, np.r_[1, -params], mode='valid') - mean
    sse = np.sum(resid**2)
    aic = 2 * k + n * np.log(sse / n)
    return aic