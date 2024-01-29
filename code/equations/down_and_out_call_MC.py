import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data.data import get_base_variables, get_exact_values
import numpy as np
from joblib import Parallel, delayed
import time

# Function to price a down-and-out call option using Monte Carlo simulation
# Source https://quant.stackexchange.com/questions/28036/black-scholes-model-for-down-and-out-european-call-option-using-monte-carlo

# Get base variables
m, r, T, sigma, S0, K, trading_days, beta, H_init, q = get_base_variables()
correct_values = get_exact_values()

def down_and_out_call_MC( m, r, T, sigma, S0, K, H, n = 5*10**6):
    
    ## Variance
    var = sigma**2
    
    np.random.seed(0)  # Set the random seed for reproducibility
    S = np.zeros((n, m))
    S[:, 0] = S0
    Z = np.random.randn(n, m - 1)
    t = np.linspace(0, T, m)
    dt = np.diff(t)
    for i in range(1, m):
        S[:, i] = S[:, i - 1] * np.exp((r - var / 2) * dt[i - 1] + np.sqrt(var * dt[i - 1]) * Z[:, i - 1])
    S_T = S[:, -1]
    payoff = np.maximum(S_T - K, 0) * (1 - np.any(S <= H, axis=1))
    price = np.exp(-r * T) * np.mean(payoff)
    return price

def main():
    h_values = range(85, 86)
    n = 8*(10**6)
    start = time.time()

    print("Number of steps ", m*n/10**9,  "Billion")
    for H in h_values:
        price = down_and_out_call_MC(m, r, T, sigma, S0, K, H, n)
        price = round(price, 4)
        print("Barrier H:",H, " ", "Price:", price, " ", "Difference:", round(price-correct_values[H],5))
    
    end = time.time()
    print("Time taken per iteration", (round((end-start)/len(h_values))), "seconds")
    
main()