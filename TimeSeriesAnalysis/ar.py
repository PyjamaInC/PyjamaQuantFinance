import numpy as np
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from estimate_orders import estimate_order, estimate_ar_params

class AR:
    def __init__(self, p=None):
        self.p = p
        self.phi = None
        self.mu = 0
        self.sigma2 = 0

    def fit(self, y, method='mle'):
        if self.p is None:
            self.p, _, _ = estimate_order(y, max_q=0)
        
        # Preprocess data
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_scaled = (y - self.y_mean) / self.y_std
        
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
        self.phi = estimate_ar_params(y, self.p)
        self.mu = np.mean(y)
        resid = y - np.convolve(y - self.mu, np.r_[1, -self.phi], mode='valid') - self.mu
        self.sigma2 = np.var(resid)

    def _fit_fiml(self, y):
        def objective(params):
            return -self._log_likelihood_fiml(y, params)

        initial_params = np.zeros(self.p + 2)  # +2 for constant and sigma2
        initial_params[0] = np.mean(y)
        initial_params[-1] = np.var(y)
        bounds = [(-0.99, 0.99)] * self.p + [(None, None), (1e-6, None)]
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
        self.phi = result.x[:-2]
        self.mu = result.x[-2]
        self.sigma2 = result.x[-1]

    def _log_likelihood_fiml(self, y, params):
        phi = params[:-2]
        mu = params[-2]
        sigma2 = params[-1]

        T = len(y)
        r = np.zeros(T)
        r[0] = sigma2 / (1 - np.sum(phi**2))
        for i in range(1, T):
            r[i] = np.sum([phi[j] * r[i-j-1] for j in range(min(i, self.p))])
        
        Sigma = toeplitz(r)
        diff = y - mu
        return -0.5 * (T * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)) + diff.T @ np.linalg.inv(Sigma) @ diff)

    def predict(self, y, n_steps):
        T = len(y)
        forecasts = np.zeros(T + n_steps)
        forecasts[:T] = y
        
        for t in range(T, T + n_steps):
            forecasts[t] = self.mu + np.sum(self.phi * (forecasts[t-self.p:t][::-1] - self.mu))
        
        return forecasts[T:]

    def get_params(self):
        return {'phi': self.phi, 'mu': self.mu, 'sigma2': self.sigma2}