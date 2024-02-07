import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from equations.adjusted_barrier import adjusted_barrier
from equations.up_and_out_call import up_and_out_call
import time

# Function to calculate percentage error
def percentage_error(price_adj, exact_price):
    return abs((price_adj - exact_price) / exact_price) * 100

# Function to find the best beta for a given set of parameters and exact price
def find_optimal_beta(S0, K, r, q, sigma, m, H, T, exact_price ):
    min_error = float('inf')
    best_beta = 0
    precision_levels=[0.01, 0.001, 0.0001]

    for precision in precision_levels:
        start = best_beta - precision if best_beta != 0 else 0
        end = best_beta + precision if best_beta != 0 else 1 + precision
        beta_values = np.arange(start, end, precision)
        
        for beta_candidate in beta_values:
            _, H_adj_up = adjusted_barrier(T, H, sigma, m, beta_candidate)
            price_adj = up_and_out_call(S0, K, T, r, q, sigma, H_adj_up)
            error = percentage_error(price_adj, exact_price)

            if error < min_error:
                min_error = error
                best_beta = beta_candidate
            
    return round(best_beta, 5), min_error

"""
start = time.time()
best_beta = find_optimal_beta(110, 100, 0.1, 0, 0.3, 50, 115, 0.2, 12.894)
print("best beta ", best_beta)

print(f"{time.time()-start} seconds")

"""


