import numpy as np
from scipy.linalg import inv, kron

def create_lag_matrix(data, p):
    n_samples, n_features = data.shape
    Y = data[p:]
    X = np.ones((n_samples - p, p * n_features + 1))
    for i in range(1, p + 1):
        X[:, (i-1)*n_features + 1:i*n_features + 1] = data[p-i:n_samples-i]
    return Y, X

def ols_estimation(Y, X):
    B = inv(X.T @ X) @ X.T @ Y
    return B

def gls_estimation(Y, X, cov_matrix):
    n_samples, n_features = Y.shape
    kron_cov = kron(np.eye(n_samples), cov_matrix)
    
    # Flatten X and Y
    Z_tilde = kron(np.eye(n_features), X)
    Y_tilde = Y.flatten(order='F')
    
    # Apply the Kronecker product transformation
    Z_T_cov_inv = np.dot(Z_tilde.T, inv(kron_cov)) 
    ZT_COV_Z = Z_T_cov_inv @ Z_tilde
    ZT_COV_Y = Z_T_cov_inv @ Y_tilde
    Beta_GLS = inv(ZT_COV_Z) @ ZT_COV_Y

    half_length = len(Beta_GLS) // 2
    first_half = Beta_GLS[:half_length]
    second_half = Beta_GLS[half_length:]

    # Combine the halves into a 2D array with 2 columns
    B_GLS = np.column_stack((first_half, second_half))

    return B_GLS

class VAR:
    def __init__(self, p):
        self.p = p
        self.coef_ = None
        self.intercept_ = None
        self.residuals_ = None
        self.cov_matrix_ = None
    
    # def fit(self, data):
    #     Y, X = create_lag_matrix(data, self.p)
    #     B = ols_estimation(Y, X)
    #     self.intercept_ = B[0]
    #     self.coef_ = B[1:].reshape((self.p, -1, data.shape[1]))
        
    #     # Calculate residuals
    #     self.residuals_ = Y - X @ B
        
    #     # Estimate covariance matrix of innovations
    #     n, m = Y.shape
    #     self.cov_matrix_ = (self.residuals_.T @ self.residuals_) / (n - self.p * m)
    
    def fit(self, data, method='OLS'):
        Y, X = create_lag_matrix(data, self.p)
        if method == 'OLS':
            B = ols_estimation(Y, X)
            self.intercept_ = B[0]
            self.coef_ = B[1:].reshape((self.p, -1, data.shape[1]))
            
            # Calculate residuals
            self.residuals_ = Y - X @ B
            n, m = Y.shape
            self.cov_matrix_ = (self.residuals_.T @ self.residuals_) / (n - self.p * m)
        elif method == 'GLS':
            self._estimate_cov_matrix(Y, X)
            B = gls_estimation(Y, X, self.cov_matrix_)
        

    def _estimate_cov_matrix(self, Y, X):
        residuals = Y - X @ ols_estimation(Y, X)
        n, m = Y.shape
        self.cov_matrix_ = (residuals.T @ residuals) / (n - self.p * m)

    def predict(self, data, steps):
        forecast = []
        current_data = data[-self.p:]
        for _ in range(steps):
            x = np.ones((1, 1 + self.p * current_data.shape[1]))
            x[0, 1:] = current_data.flatten()
            y_hat = x @ np.vstack([self.intercept_, self.coef_.reshape(-1, current_data.shape[1])])
            forecast.append(y_hat[0])
            current_data = np.vstack([current_data[1:], y_hat])
        return np.array(forecast)
    
    def fit_predict(self, data, steps):
        self.fit(data)
        return self.predict(data, steps)
