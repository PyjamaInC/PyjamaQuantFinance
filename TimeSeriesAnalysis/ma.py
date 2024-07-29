import numpy as np
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from estimate_orders import estimate_order, estimate_ma_params

class MA:
    def __init__(self, q=None):
        self.q = q
        self.theta = None
        self.mu = 0
        self.sigma2 = 0

    def fit(self, y, method='mle'):
        if self.q is None:
            _, _, self.q = estimate_order(y, max_p=0)
        
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
        self.theta = estimate_ma_params(y, self.q)
        self.mu = np.mean(y)
        resid = y - np.convolve(y - self.mu, np.r_[1, self.theta], mode='valid') - self.mu
        self.sigma2 = np.var(resid)

    def _fit_fiml(self, y):
        def objective(params):
            return -self._log_likelihood_fiml(y, params)

        initial_params = np.zeros(self.q + 2)  # +2 for constant and sigma2
        initial_params[0] = np.mean(y)
        initial_params[-1] = np.var(y)
        bounds = [(-0.99, 0.99)] * self.q + [(None, None), (1e-6, None)]
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
        self.theta = result.x[:-2]
        self.mu = result.x[-2]
        self.sigma2 = result.x[-1]

    def _log_likelihood_fiml(self, y, params):
        theta = params[:-2]
        mu = params[-2]
        sigma2 = params[-1]

        T = len(y)
        r = np.zeros(T)
        r[0] = sigma2 * (1 + np.sum(theta**2))
        for i in range(1, min(self.q + 1, T)):
            r[i] = sigma2 * theta[i-1]
        
        Sigma = toeplitz(r)
        diff = y - mu
        return -0.5 * (T * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)) + diff.T @ np.linalg.inv(Sigma) @ diff)

    def predict(self, y, n_steps):
        T = len(y)
        forecasts = np.zeros(n_steps)
        residuals = np.zeros(T)

        for t in range(self.q, T):
            yhat = self.mu + np.dot(self.theta, residuals[t-self.q:t][::-1])
            residuals[t] = y[t] - yhat

        for h in range(1, n_steps + 1):
            if h <= self.q:
                forecasts[h-1] = self.mu + np.dot(self.theta[:self.q-h+1], residuals[-(self.q-h+1):][::-1])
            else:
                forecasts[h-1] = self.mu
        
        return forecasts

    def get_params(self):
        return {'theta': self.theta, 'mu': self.mu, 'sigma2': self.sigma2}