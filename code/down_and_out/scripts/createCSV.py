import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier
from equations.down_and_out_call_Brown import price_down_and_out_call_brown
from generate_data.find_beta import find_optimal_beta
import pandas as pd
import numpy as np
import time

# Function to compute prices for a given set of parameters
def compute_prices(S0, K, r, m, T, H, sigma, trading_days, n):
    H_adj_down, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)
    price_iter = price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
    price = down_and_call_book(S0, K, T, r, q, sigma, H, H, H)
    price_adj = down_and_call_book(S0, K, T, r, q, sigma, H, H_adj_down, H_adj_up)
    error_percent = ((price_iter - price_adj)/price_iter)*100
    H_percent = ((S0 - H)/S0)*100
    
    return [S0, K, r, m, T, H, sigma, trading_days, price_iter, price, price_adj, error_percent, H_percent]

# Values
S0 = 100
K = 100 
q = 0
m = 50
trading_days = 250
h_min = round(0.9*S0)
h_max = S0
beta = 0.5826
t_values = np.arange(0.2, 5.1, 0.1)
sigma_values = np.arange(0.2, 0.61, 0.05)
h_values = range(h_min, h_max)
r_values = [0.1]

n = 3*10**7

# Calculate total iterations
total_iterations = len(r_values) * len(h_values) * len(t_values) * len(sigma_values)
current_iteration = 0

# Start the timer
start_time = time.time()

# Non-parallel computation with progress tracking
results = []
for r in r_values:
    for H in h_values:
        for T in t_values:
            T_rounded = round(T, 1)
            for sigma in sigma_values:
                result = compute_prices(S0, K, r, m, T_rounded, H, sigma, trading_days, n)
                results.append(result)
                current_iteration += 1
                print(f"Progress: {current_iteration}/{total_iterations} iterations completed.", end='\r', flush=True)

# Move to the next line after the loop is complete
print()

# Create DataFrame from results
df = pd.DataFrame(results, columns=['S0', 'K', 'r', 'm', 'T', 'H', 'sigma', 'trading_days', 'price_iter', 'price', 'price_adj','error_percent', 'H_percent' ])

# Rounding the values to three decimal places
df = df.round(4)

# Save the DataFrame to a CSV file
df.to_csv('accurate_data.csv', index=False)

# Print elapsed time
elapsed_time = time.time() - start_time
print(f"Simulation completed in {elapsed_time:.2f} seconds")
print("Simulation data saved to 'data_K200s.csv'")
