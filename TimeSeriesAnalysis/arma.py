import numpy as np
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from estimate_orders import estimate_order, estimate_arma_params

class ARMA:
    def __init__(self, p=None, q=None):
        self.p = p
        self.q = q
        self.d = 0
        self.phi = None
        self.theta = None
        self.mu = 0
        self.sigma2 = 0

    def fit(self, y, method='mle'):
        if self.p is None or self.q is None:
            self.p, self.d, self.q = estimate_order(y)
        
        # Apply differencing
        diff_y = np.diff(y, n=self.d) if self.d > 0 else y
        
        # Preprocess data
        self.y_mean = np.mean(diff_y)
        self.y_std = np.std(diff_y)
        y_scaled = (diff_y - self.y_mean) / self.y_std
        
        if method.lower() == 'mle':
            self._fit_mle(y_scaled)
        elif method.lower() == 'fiml':
            self._fit_fiml(y_scaled)
        else:
            raise ValueError("Method must be either 'mle' or 'fiml'")
        
        # Rescale parameters
        self.mu = self.y_mean + self.y_std * self.mu
        self.sigma2 *= self.y_std**2

    def _fit_mle(self, y):
        self.phi, self.theta = estimate_arma_params(y, self.p, self.q)
        self.mu = np.mean(y)
        resid = y - np.convolve(y - self.mu, np.r_[1, -self.phi], mode='valid') - self.mu
        resid = resid - np.convolve(resid, np.r_[1, self.theta], mode='valid')
        self.sigma2 = np.var(resid)

    def _fit_fiml(self, y):
        def objective(params):
            return -self._log_likelihood_fiml(y, params)

        initial_params = np.zeros(self.p + self.q + 2)  # +2 for constant and sigma2
        initial_params[0] = np.mean(y)
        initial_params[-1] = np.var(y)
        bounds = [(-0.99, 0.99)] * (self.p + self.q) + [(None, None), (1e-6, None)]
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
        self.phi = result.x[:self.p]
        self.theta = result.x[self.p:-2]
        self.mu = result.x[-2]
        self.sigma2 = result.x[-1]

    def _log_likelihood_fiml(self, y, params):
        phi = params[:self.p]
        theta = params[self.p:-2]
        mu = params[-2]
        sigma2 = params[-1]

        T = len(y)
        r = self._compute_acf(phi, theta, sigma2, T)
        
        # Use Innovations Algorithm for more efficient likelihood computation
        v = np.zeros(T)
        K = np.zeros((T, T))
        
        v[0] = r[0]
        K[1, 0] = r[1] / v[0]
        
        for t in range(1, T):
            v[t] = r[0] - np.sum(K[t, :t]**2 * v[:t])
            if t < T-1:
                K[t+1, :t+1] = (r[1:t+2][::-1] - np.sum(K[1:t+1, :t] * K[t+1, :t] * v[:t], axis=1)) / v[t]
        
        eps = y - mu
        for t in range(1, T):
            eps[t] -= np.dot(K[t, :t], eps[:t][::-1])
        
        return -0.5 * (T * np.log(2 * np.pi) + np.sum(np.log(v)) + np.sum(eps**2 / v))

    def _compute_acf(self, phi, theta, sigma2, T):
        ar = np.r_[1, -phi]
        ma = np.r_[1, theta]
        arma = np.polymul(ar, ma)
        
        r = np.zeros(T)
        r[0] = sigma2 * np.sum(ma**2) / np.sum(ar**2)
        
        for k in range(1, T):
            r[k] = np.sum(arma[k:] * arma[:-k] * r[0])
        
        return r

    def predict(self, y, n_steps):
        # First, apply differencing to the input series
        diff_y = np.diff(y, n=self.d) if self.d > 0 else y
        
        T = len(diff_y)
        forecasts = np.zeros(n_steps)
        residuals = np.zeros(T + n_steps)
        
        for t in range(max(self.p, self.q), T):
            yhat = self.mu + np.dot(self.phi, diff_y[t-self.p:t][::-1] - self.mu) + np.dot(self.theta, residuals[t-self.q:t][::-1])
            residuals[t] = diff_y[t] - yhat

        y_extended = np.concatenate([diff_y, forecasts])
        for t in range(T, T + n_steps):
            forecasts[t-T] = self.mu + np.dot(self.phi, y_extended[t-self.p:t][::-1] - self.mu) + np.dot(self.theta, residuals[t-self.q:t][::-1])
            y_extended[t] = forecasts[t-T]

        # If differencing was applied, we need to integrate the forecasts
        if self.d > 0:
            cumsum_forecasts = np.cumsum(forecasts)
            integrated_forecasts = y[-1] + cumsum_forecasts
            return integrated_forecasts
        else:
            return forecasts

    def get_params(self):
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'phi': self.phi,
            'theta': self.theta,
            'mu': self.mu,
            'sigma2': self.sigma2
        }