import numpy as np
from scipy import stats

def test():
  # Example usage:
  m = 50    # number of time steps
  r = 0.1   # risk-free rate
  T = 0.2      # time to maturity
  sigma = 0.3 # variance of the underlying asset
  S0 = 100   # initial stock price
  K = 100    # strike price
  H = 85
  q = 0
  confidence_level=0.95
    
  n_paths = 10**6 # Number of simulation paths
  n_steps = m  # Number of time steps
  dt = T / n_steps  # Time step size

  dW = np.sqrt(dt) * np.random.randn(n_steps, n_paths)
  S = np.zeros((n_steps + 1, n_paths))
  S[0, :] = S0

  for i in range(1, n_steps + 1):
      S[i, :] = S[i - 1, :] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * dW[i - 1, :])
      S[i, :] = np.where(S[i, :] < H, 0, S[i, :])  # Apply barrier condition

  # Calculate payoffs
  payoffs = np.maximum(S[-1, :] - K, 0) * np.exp(-r * T)
  payoffs = np.where(S.any(axis=0), payoffs, 0)

  # Calculate option price and standard error
  option_price = np.mean(payoffs)
  std_error = np.std(payoffs)
  sem = std_error / np.sqrt(n_paths)

  # Calculate confidence interval
  z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
  confidence_interval = z_score * sem

  return [option_price, sem, confidence_interval]    




value = test()
print(value)