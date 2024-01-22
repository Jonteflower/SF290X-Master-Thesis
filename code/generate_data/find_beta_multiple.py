import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier
from data import get_base_variables

# Get base variables
m, r, T, sigma, S0, K, trading_days, beta, H_init, q = get_base_variables()

# Read the existing training data
df = pd.read_csv('data.csv')
df = df[(df['H'] <= 92)]

# Function to calculate percentage error
def percentage_error(price_adj, price_mc):
    return abs((price_adj - price_mc) / price_mc) * 100

# Function to find the best beta for a given H
def find_best_beta_for_H(H, df):
    min_error = float('inf')
    best_beta = None
    df_H = df[df['H'] == H]

    # First iteration with larger step size
    beta_values_large_step = np.arange(0.0, 1.0001, 0.1)
    for beta_candidate in beta_values_large_step:
        total_error = 0
        for index, row in df_H.iterrows():
            H_adj_down, H_adj_up = adjusted_barrier(row['T'], row['H'], row['sigma'], m, beta_candidate)
            price_adj = down_and_call_book(S0, K, row['T'], r, q, row['sigma'], row['H'], H_adj_down, H_adj_up)
            error = percentage_error(price_adj, row['price_mc'])
            total_error += error
        average_error = total_error / len(df_H)
        if average_error < min_error:
            min_error = average_error
            best_beta = beta_candidate

    # Second iteration with smaller step size
    beta_values_small_step = np.arange(best_beta - 0.1, best_beta + 0.1001, 0.0001)
    for beta_candidate in beta_values_small_step:
        total_error = 0
        for index, row in df_H.iterrows():
            H_adj_down, H_adj_up = adjusted_barrier(row['T'], row['H'], row['sigma'], m, beta_candidate)
            price_adj = down_and_call_book(S0, K, row['T'], r, q, row['sigma'], row['H'], H_adj_down, H_adj_up)
            error = percentage_error(price_adj, row['price_mc'])
            total_error += error
        average_error = total_error / len(df_H)
        if average_error < min_error:
            min_error = average_error
            best_beta = beta_candidate

    return H, best_beta, min_error

# Parallel computation for each unique H
optimal_betas = Parallel(n_jobs=-1)(delayed(find_best_beta_for_H)(H, df) for H in df['H'].unique())

# Convert results to a dictionary
optimal_beta_dict = {H: {'beta': beta, 'error': error} for H, beta, error in optimal_betas}

# Print the best beta for each H
for H, data in optimal_beta_dict.items():
    print(f"H: {H}, Best Beta: {data['beta']}, with an average error of: {data['error']}")
