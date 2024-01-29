"""
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data.data import get_base_variables, get_exact_values

def price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n=10**7):
    n_paths = n  # Number of simulation paths
    n_steps = m     # Number of time steps
    dt = T / n_steps # Step size of the time step

    dW = np.sqrt(dt) * np.random.randn(n_steps, n_paths)
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    for i in range(1, n_steps + 1):
        S[i, :] = S[i - 1, :] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * dW[i - 1, :])
        # Check for barrier
        S[i, :] = np.where(S[i, :] < H, 0, S[i, :])

    # Calculate payoffs
    payoffs = np.maximum(S[-1, :] - K, 0)
    # Only consider paths that did not hit the barrier
    payoffs = np.where(S.any(axis=0), payoffs, 0)
    option_price = np.mean(payoffs) * np.exp(-r * T)  # Discounting to present value
    return option_price


def main(): 
    # Intial values
    h_values = range(85, 86)
    n = (8*10**6)
    start = time.time()
    
    # Get base variables
    m, r, T, sigma, S0, K, trading_days, beta, H, q = get_base_variables()
    correct_values = get_exact_values()

    # Print the number of steps
    print("Number of steps ", m*n/10**9,  "Billion")
    
    # Start time
    start = time.time()
    
    # Loop all H
    for H in h_values:
        price = price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
        
        print("Barrier H:",H, " ", "Price:", round(price, 3), " ", "Difference:", round(price-correct_values[H],4))
    
    #Print results
    time_per_iteration = round((time.time()-start)/len(h_values), 1)
    
    print( "Time/iteration:", time_per_iteration, "s" )
    
main()

"""
