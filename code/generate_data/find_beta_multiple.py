import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier

from data import get_base_variables

# Get base variables
m, r, T, sigma, S0, K, trading_days, beta, H_init, q = get_base_variables()

# Read the existing training data
df = pd.read_csv('data.csv')
df = df[(df['H'] >= 89) & (df['H'] <= 94) ]

# Define the beta range
#beta_values = np.arange(0.0, 1.0001, 0.0001)  # Adjust the range as needed
beta_values = np.arange(0.58, 0.70001, 0.0001)  # Adjust the range as needed

# Function to calculate percentage error
def percentage_error(price_adj, price_mc):
    return abs((price_adj - price_mc) / price_mc) * 100

# Initialize a dictionary to store the best beta for each H
optimal_betas = {}

# Iterate over each unique H value
for H in df['H'].unique():
    min_error = float('inf')
    best_beta = None
    df_H = df[df['H'] == H]

    # Iterate over beta values
    for beta_candidate in beta_values:
        total_error = 0

        # Loop through rows with the same H value
        for index, row in df_H.iterrows():
            # Calculate the adjusted price with the current beta_candidate
            H_adj_down, H_adj_up = adjusted_barrier(row['T'], row['H'], row['sigma'], m, beta_candidate)
            price_adj = down_and_call_book(S0, K, row['T'], r, q, row['sigma'], row['H'], H_adj_down, H_adj_up)

            # Calculate the percentage error
            error = percentage_error(price_adj, row['price_mc'])
            total_error += error

        # Calculate average error for this beta_candidate
        average_error = total_error / len(df_H)

        # Check if this is the best beta for this H
        if average_error < min_error:
            min_error = average_error
            best_beta = beta_candidate

    # Store the best beta and its error for this H
    optimal_betas[H] = {'beta': best_beta, 'error': min_error}

# Print the best beta for each H
for H, data in optimal_betas.items():
    print(f"H: {H}, Best Beta: {data['beta']}, with an average error of: {data['error']}")
