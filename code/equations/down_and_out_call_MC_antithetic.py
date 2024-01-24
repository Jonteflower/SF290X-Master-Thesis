import numpy as np
from joblib import Parallel, delayed
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_data.data import get_base_variables, get_exact_values

# Function to price a down-and-out call option using Monte Carlo simulation
# Source https://quant.stackexchange.com/questions/28036/black-scholes-model-for-down-and-out-european-call-option-using-monte-carlo

# Get base variables
m, r, T, sigma, S0, K, trading_days, beta, H_init, q = get_base_variables()
correct_values = get_exact_values()

def down_and_out_call_MC_antithetic(m, r, T, sigma, S0, K, H, n=5*10**6):
    ## Variance
    var = sigma**2
    
    np.random.seed(0)  # Set the random seed for reproducibility
    S = np.zeros((n, m))  # Matrix to store stock paths
    S[:, 0] = S0
    Z = np.random.randn(n, m - 1)  # Random variables for path generation
    t = np.linspace(0, T, m)  # Time points
    dt = np.diff(t)  # Differences in time points
    
    # Generate stock paths using the standard Monte Carlo and antithetic variates
    for i in range(1, m):
        S[:, i] = S[:, i - 1] * np.exp((r - var / 2) * dt[i - 1] + np.sqrt(var * dt[i - 1]) * Z[:, i - 1])
    
    # Antithetic stock paths
    S_antithetic = np.zeros((n, m))
    S_antithetic[:, 0] = S0
    for i in range(1, m):
        S_antithetic[:, i] = S_antithetic[:, i - 1] * np.exp((r - var / 2) * dt[i - 1] - np.sqrt(var * dt[i - 1]) * Z[:, i - 1])
    
    # Calculate the payoff for standard and antithetic paths
    S_T = S[:, -1]
    S_T_antithetic = S_antithetic[:, -1]
    payoff = np.maximum(S_T - K, 0) * (1 - np.any(S <= H, axis=1))
    payoff_antithetic = np.maximum(S_T_antithetic - K, 0) * (1 - np.any(S_antithetic <= H, axis=1))
    
    # Combine payoffs from both paths and calculate the price
    payoff_combined = (payoff + payoff_antithetic) / 2
    price = np.exp(-r * T) * np.mean(payoff_combined)
    
    return price


def main():
    
    h_values = range(85, 87)
    n = 6*(10**6)
    start = time.time()

    # Print the number of steps
    print("Number of steps ", m*n/10**9,  "Billion")
    
    for H in h_values:
        price = down_and_out_call_MC_antithetic(m, r, T, sigma, S0, K, H, n)

        print("Barrier H:",H, " ", "Price:", round(price, 4), " ", "Difference:", round(price-correct_values[H],5))
    
    #Print the end time
    end = time.time()
    print("Time taken per iteration", (round((end-start)/len(h_values))), "seconds")
    
#main()