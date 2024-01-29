import numpy as np
import time
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data.data import get_base_variables, get_exact_values

def price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n=10**7, confidence_level=0.95):
    n_paths = n  # Number of simulation paths
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

    return option_price, sem, confidence_interval

def main(): 
    # Initial values
    h_values = range(85, 86)
    n = 2 * 10**7
    
    # Get base variables
    m, r, T, sigma, S0, K, trading_days, beta, H, q = get_base_variables()
    correct_values = get_exact_values()

    # Print the number of steps
    print("Number of steps:", m * n / 10**9, "Billion")
    
    # Start time
    start = time.time()
    
    # Loop all H
    for H in h_values:
        price, sem, conf_interval = price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
        lower_bound = price - conf_interval
        upper_bound = price + conf_interval
        print("Barrier H:", H, "Price:", round(price, 3), "SEM:", round(sem, 3), "Confidence Interval:", f"({round(lower_bound, 3)}, {round(upper_bound, 3)})", "Difference:", round(price - correct_values[H], 4))
    
    # Print results
    time_per_iteration = round((time.time() - start) / len(h_values), 1)
    print("Time/iteration:", time_per_iteration, "s")
    

#main()