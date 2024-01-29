import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier
from data import get_base_variables

# Function to calculate percentage error
def percentage_error(price_adj, price_mc):
    return abs((price_adj - price_mc) / price_mc) * 100

# Function to find the best beta
def find_optimal_beta(df, S0, K, r, q, sigma, m, H, precision_levels=[0.1, 0.01, 0.001]):
    min_error = float('inf')
    best_beta = 0

    for precision in precision_levels:
        start = best_beta - precision if best_beta != 0 else 0
        end = best_beta + precision if best_beta != 0 else 1 + precision
        beta_values = np.arange(start, end, precision)

        for beta_candidate in beta_values:
            total_error = 0
            df_H = df[df['H'] == H]
            for _, row in df_H.iterrows():
                H_adj_down, H_adj_up = adjusted_barrier(row['T'], H, row['sigma'], m, beta_candidate)
                price_adj = down_and_call_book(S0, K, row['T'], r, q, row['sigma'], H, H_adj_down, H_adj_up)
                error = percentage_error(price_adj, row['price_mc'])
                total_error += error

            average_error = total_error / len(df_H)
            if average_error < min_error:
                min_error = average_error
                best_beta = beta_candidate

    return H, best_beta, min_error

# Main execution block
def main():
    m, r, T, sigma, S0, K, trading_days, beta, H_init, q = get_base_variables()

    # Read the existing training data
    df = pd.read_csv('data.csv')
    df = df[(df['sigma'] == sigma)]

    # Parallel computation for each unique H
    optimal_betas = Parallel(n_jobs=-1)(delayed(find_optimal_beta)(df, S0, K, r, q, sigma, m, H) for H in df['H'].unique())

    # Convert results to a dictionary
    optimal_beta_dict = {H: {'beta': beta, 'error': error} for H, beta, error in optimal_betas}

    # Print the best beta for each H
    print("For sigma value of ", df['sigma'].unique())
    for H, data in optimal_beta_dict.items():
        print(f"H: {H}, Best Beta: {data['beta']}, with an average error of: {data['error']}")

if __name__ == "__main__":
    main()
