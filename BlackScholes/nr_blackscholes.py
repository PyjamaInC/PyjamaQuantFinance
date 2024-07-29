
import numpy as np
import scipy.stats as st

# A simple naive implementation of Newton Raphson algorithm for calculating implied volatility of black scholes model

def Black_Scholes(Option_Type, S_0, K, sigma, tau, r):
    
    d1 = (np.log(S_0 / K) + (r + 0.5*np.power(sigma, 2.0)) * (tau)) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if str(Option_Type).lower() == "c" or str(Option_Type) == "1":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif str(Option_Type).lower() == "p" or str(Option_Type) == "-1":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S_0
    return value
    
def ImpliedVolatility(Option_Type, S_0, K, sigma, tau, r, V_m):
    error = 1e10

    optPrice = lambda sigma: Black_Scholes(Option_Type, S_0, K, sigma, tau, r)
    vega = lambda sigma: dV_on_sigma(S_0, K, sigma, tau, r)

    while error > 10e-10:
        f = V_m - optPrice(sigma)
        f_prim = -vega(sigma)
        sigma_new = sigma - f / f_prim

        error = abs(sigma_new - sigma)
        sigma = sigma_new

    return sigma 

def dV_on_sigma(S_0, K, sigma, tau, r):

    # Mathematical modeling p.83 chapter 4
    d2 = (np.log(S_0 / float(K)) + (r - 0.5 * np.power(sigma, 2.0)) * tau) / float(sigma * np.sqrt(tau))
    val = K * np.exp(-r * tau) * st.norm.pdf(d2) * np.sqrt(tau)
    return val
