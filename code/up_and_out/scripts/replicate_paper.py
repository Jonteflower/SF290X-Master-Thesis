import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import time
from equations.up_and_out_call import up_and_out_call
from equations.adjusted_barrier import adjusted_barrier, adjusted_barrier_custom
from equations.up_and_out_call_Brown import price_up_and_out_call_brown
from generate_data.find_beta import find_optimal_beta


def compute_prices(S0, K, r, m, T, H, sigma, trading_days, n):
    # Adjust the barriers
    H_adj_down, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)

    # Calculate the prices
    price = up_and_out_call(S0, K, T, r, q, sigma, H, H, H)
    price_adj = up_and_out_call(S0, K, T, r, q, sigma, H, H_adj_down, H_adj_up)
    
    #Exact price
    price_iter = price_up_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
    price_iter = price_iter[0]  # Assuming this returns a list or tuple, take the first element
    
    #Best beta value
    best_beta = find_optimal_beta(S0, K, r, q, sigma, m, H, T, price_iter)
    
    return [S0, K, r, m, T, H, sigma, trading_days, price_iter, price, price_adj, best_beta[0]]


# Constants
q = 0
trading_days = 250
beta = 0.5826
T = 1  # Specified T constant
r = 0.1
sigma_values = [0.35, 0.4]
n = 2*10**7
file_name = 'paper_values.csv'
m=50
K=300
S0=300 
h_min = int(round(0.85 * S0))
h_max = int(S0)
h_values = list(range(h_min, h_max + 1))  # Ensure inclusive range
    
# Check if the combined file exists, and if so, load it
existing_df = pd.DataFrame()

# Loop through each K value and compute prices if the combination is new
for sigma in sigma_values:
    # Calculate total iterations for the progress bar (assuming h_values is the only iterator)
    total_iterations = len(h_values)
    current_iteration = 0

    # Start the timer
    start_time = time.time()

    for H in h_values:
        result = compute_prices(S0, K, r, m, T, H, sigma, trading_days, n)
        # Append result to CSV
        with open(file_name, 'a') as f:
            pd.DataFrame([result], columns=['S0', 'K', 'r', 'm', 'T', 'H', 'sigma', 'trading_days', 'price_iter', 'price', 'price_adj', 'best_beta']).to_csv(f, mode='a', header=f.tell()==0, index=False)
        
        current_iteration += 1
        print(f"Progress: {current_iteration}/{total_iterations} iterations completed.", end='\r', flush=True)
    
    # Print elapsed time for current K
    elapsed_time = time.time() - start_time
    print(f"\nSimulation for K={K} completed in {elapsed_time:.2f} seconds")

print(f"Simulation data saved to {file_name}")