import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import time
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier, adjusted_barrier_custom
from equations.down_and_out_call_Brown import price_down_and_out_call_brown
from generate_data.find_beta import find_optimal_beta


def compute_prices(S0, K, r, m, T, H, sigma, trading_days, n, data):
    
    # Adjust the barriers
    H_adj_down, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)
    H_adj_down_cust, H_adj_up_cust = adjusted_barrier_custom(T, H,S0, K, sigma, m, beta, data)

    # Calculate the prices
    price_adj = down_and_call_book(S0, K, T, r, q, sigma, H, H_adj_down, H_adj_up)
    price_adj_cust = down_and_call_book(S0, K, T, r, q, sigma, H, H_adj_down_cust, H_adj_up_cust)

    #Exact price
    price_iter = price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
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

for H in h_values:
    S0, K, r, m, T, H, sigma, trading_days, price_iter, price, price_adj, best_beta = compute_prices(S0, K, r, m, T, H, sigma, trading_days, n)
    
    

