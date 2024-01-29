import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import time
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier
from equations.down_and_out_call_Brown import price_down_and_out_call_brown
# from equations.down_and_out_call_MC import down_and_out_call_MC

# Function to compute prices for a given set of parameters
def compute_prices(S0, K, r, m, T, H, sigma, trading_days):
    H_adj_down, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)
    #price_mc = down_and_out_call_MC(m, r, T, sigma, S0, K, H)
    price_iter = price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q)
    price = down_and_call_book(S0, K, T, r, q, sigma, H, H, H)
    price_adj = down_and_call_book(S0, K, T, r, q, sigma, H, H_adj_down, H_adj_up)
    return [S0, K, r, m, T, H, sigma, trading_days, price_iter, price, price_adj]

# Values
S0 = 100
K = 100 
q = 0
m = 50
trading_days = 250
h_min = 90
h_max = S0
beta = 0.5826
t_values = np.arange(0.2, 5.1, 0.1)
sigma_values = np.arange(0.2, 0.61, 0.05)
h_values = range(h_min, h_max)
r_values = np.arange(0, 0.15, 0.05)

# Start the timer
start_time = time.time()

# Parallel computation, change n_jobs for number of cores
results = Parallel(n_jobs=-1)(delayed(compute_prices)(S0, K, r, m, T_rounded, H, sigma, trading_days)
                              for r in r_values
                              for H in h_values
                              for T in t_values
                              for sigma in sigma_values
                              for T_rounded in [round(T, 1)])

# Create DataFrame from results
df = pd.DataFrame(results, columns=['S0', 'K', 'r', 'm', 'T', 'H', 'sigma', 'trading_days', 'price_mc', 'price', 'price_adj'])

# Rounding the values to three decimal places
df = df.round(4)

# Save the DataFrame to a CSV file
df.to_csv('simulation_results.csv', index=False)

# Print elapsed time
elapsed_time = time.time() - start_time
print(f"Simulation completed in {elapsed_time:.2f} seconds")
print("Simulation data saved to 'simulation_results.csv'")
