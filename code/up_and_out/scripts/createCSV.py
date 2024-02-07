import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equations.up_and_out_call import up_and_out_call
from equations.adjusted_barrier import adjusted_barrier
from equations.up_and_out_call_Brown import price_up_and_out_call_brown
from generate_data.find_beta import find_optimal_beta
import pandas as pd
import numpy as np
import time

# Function to compute prices for a given set of parameters
def compute_prices(S0, K, r, m, T, H, sigma, trading_days, n):
    _, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)
    price_iter = price_up_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
    price = up_and_out_call(S0, K, T, r, q, sigma, H)
    price_adj = up_and_out_call(S0, K, T, r, q, sigma, H_adj_up)
    H_percent = ((S0 - H)/S0)*100
    #Best beta value
    best_beta = find_optimal_beta(S0, K, r, q, sigma, m, H, T, price_iter[0])
    print(best_beta)
    return [S0, K, r, m, T, H, sigma, trading_days, price_iter, price, price_adj, H_percent, best_beta[0]]

# Values
S0 = 110
K = 100 
q = 0
m = 50
trading_days = 250
h_min = 115
h_max = 155
beta = 0.5826
t_values = np.arange(0.2, 5.3, 0.4)
sigma_values = np.arange(0.2, 0.61, 0.1)
h_values = range(h_min, h_max)
r_values = [0.1]
n = 1*10**7

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
df = pd.DataFrame(results, columns=['S0', 'K', 'r', 'm', 'T', 'H', 'sigma', 'trading_days', 'price_iter', 'price', 'price_adj', 'H_percent'])

# Rounding the values to three decimal places
df = df.round(4)

# Save the DataFrame to a CSV file
df.to_csv('data_up_out_1.csv', index=False)

# Print elapsed time
elapsed_time = time.time() - start_time
print(f"Simulation completed in {elapsed_time:.2f} seconds")
print("Simulation data saved to 'data_K200s.csv'")
