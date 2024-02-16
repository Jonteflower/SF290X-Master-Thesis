import numpy as np
from scipy.stats import norm

def get_parameter_values(S0, K, T, r, q, sigma): 
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    lambda_ = (r - q + 0.5 * sigma**2) / sigma**2
    c = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) 
    
    return lambda_, c

def up_and_out_call(S0, K, T, r, q, sigma, H):
    lambda_, c = get_parameter_values(S0, K, T, r, q, sigma)
    
    y = np.log(H**2 / (S0*K)) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    x1 = np.log(S0/H) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    y1 = np.log(H/S0) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)

    cui = 0
    cuo = 0
    
    # Check if barrier is greater than the strike price
    if H > K:
        cui = (
            S0 * norm.cdf(x1) * np.exp(-q * T)
            - K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
            - S0 * np.exp(-q * T) * (H/S0)**(2*lambda_) * (norm.cdf(-y) - norm.cdf(-y1))
            + K * np.exp(-r*T) * (H/S0)**(2*lambda_ - 2)  * (norm.cdf(-y + sigma * np.sqrt(T)) - norm.cdf(-y1 + sigma * np.sqrt(T)))
        )
        cuo = c - cui
    else:
        # When the barrier is less than or equal to the strike price,
        # the up-and-out call value is zero
        cuo = 0
        cui = c

    return cui, cuo
    

# Example usage:
S0 = 110  # Current stock price
K = 100   # Strike price
T = 0.2   # Time to maturity in years
r = 0.1   # Risk-free interest rate
q = 0.0   # Dividend yield
sigma = 0.3  # Volatility
H = 110    # Barrier
m = 50
delta_T = T / m
beta = 0.5826
# Calculate up-and-out call price

H_adjusted = H * np.exp( beta * sigma * np.sqrt(delta_T))
cui, cuo = up_and_out_call(S0, K, T, r, q, sigma, H_adjusted)
#print(cui, cuo)

