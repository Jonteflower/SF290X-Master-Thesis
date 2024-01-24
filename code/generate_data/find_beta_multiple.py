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
m, r, _, _, S0, K, trading_days, _, H_init, q = get_base_variables()

# Read the existing training data
df = pd.read_csv('data.csv')

# Function to calculate percentage error
def percentage_error(price_adj, price_mc):
    return abs((price_adj - price_mc) / price_mc) * 100

# Function to find the best beta for a given H
def find_best_beta_for_H(H, df_T_sigma):
    min_error = float('inf')
    best_beta = None

    # Filter dataframe for the current H value
    df_H = df_T_sigma[df_T_sigma['H'] == H]

    # First iteration with larger step size
    beta_values_large_step = np.arange(0.0, 1.0001, 0.1)
    for beta_candidate in beta_values_large_step:
        total_error = 0
        for _, row in df_H.iterrows():
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
        for _, row in df_H.iterrows():
            H_adj_down, H_adj_up = adjusted_barrier(row['T'], row['H'], row['sigma'], m, beta_candidate)
            price_adj = down_and_call_book(S0, K, row['T'], r, q, row['sigma'], row['H'], H_adj_down, H_adj_up)
            error = percentage_error(price_adj, row['price_mc'])
            total_error += error
        average_error = total_error / len(df_H)
        if average_error < min_error:
            min_error = average_error
            best_beta = beta_candidate

    return H, best_beta, min_error

# Initialize a list to store the results
results = []

# Iterate over each unique combination of T and sigma
for T in df['T'].unique():
    for sigma in df['sigma'].unique():
        df_T_sigma = df[(df['T'] == T) & (df['sigma'] == sigma)]

        # Parallel computation for each unique H
        optimal_betas = Parallel(n_jobs=5)(delayed(find_best_beta_for_H)(H, df_T_sigma) for H in df_T_sigma['H'].unique())

        # Append results
        for H, beta, error in optimal_betas:
            results.append({'T': T, 'sigma': sigma, 'H': H, 'Best Beta': beta, 'Average Error': error})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv('Beta_values.csv', index=False)

print("Results saved to 'Beta_values.csv'")
