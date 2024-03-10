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
    precision_interval = 0.1
    precision_levels=[precision_interval/10, precision_interval/100, precision_interval/10000, precision_interval/100000]

    for precision in precision_levels:
        start = best_beta - precision if best_beta != 0 else 0
        end = best_beta + precision if best_beta != 0 else 1 + precision
        beta_values = np.arange(start, end, precision)
        
        print(beta_values)
        for beta_candidate in beta_values:
            _, H_adj_up = adjusted_barrier(T, H, sigma, m, beta_candidate)
            price_adj = up_and_out_call(S0, K, T, r, q, sigma, H_adj_up)
            error = percentage_error(price_adj, exact_price)

            #print("Price adjusted: ", price_adj, "exact price ", exact_price)
            
            if round(error, 4) <= min_error:
                min_error = error
                best_beta = beta_candidate
            
    return round(best_beta, 5), min_error

"""


"""

T = 2
sigma = 0.4
m = 50
r = 0.1
S0 = 110
K = 100
q = 0
H = 111
exact_price = 0.01085328855903971
beta_ = 0.5826

start = time.time()
best_beta = find_optimal_beta(S0, K, r, q, sigma, m, H, T,  exact_price)

### Standard beta  
_, H_adj_up_1 = adjusted_barrier(T, H, sigma, m, beta_)
price_adj_1 = up_and_out_call(S0, K, T, r, q, sigma, H_adj_up_1)
print("Regular price adjusted ",price_adj_1 )

## Best beta 
_, H_adj_up = adjusted_barrier(T, H, sigma, m, best_beta[0])
price_adj = up_and_out_call(S0, K, T, r, q, sigma, H_adj_up)
print("Best beta price adjusted ",price_adj )

print("best beta ", best_beta[0])
print("H_adj_up ", H_adj_up, "H_adj_up_1",H_adj_up_1)
print(f"{time.time()-start} seconds")


