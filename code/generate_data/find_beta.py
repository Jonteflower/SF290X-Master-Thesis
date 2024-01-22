import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier
from equations.down_and_out_call_MC import down_and_out_call_MC
from data import get_base_variables

# Get base variables
m, r, T, sigma, S0, K, trading_days, beta, H_init, q = get_base_variables()

# Read the existing training data
df = pd.read_csv('data.csv')
df = df[df['H'] == 94]

# Define the beta range
beta_values = np.arange(0.6, 0.80001, 0.0001)  # Adjust the range as needed

# Variables to store the best beta and its error
best_beta = None
min_error = float('inf')

# Function to calculate percentage error
def percentage_error(price_adj, price_mc):
    return abs((price_adj - price_mc) / price_mc) * 100

# Total number of iterations for the progress bar
total_iterations = len(beta_values) * len(df)
current_iteration = 0

# Iterate over beta values
for beta_candidate in beta_values:
    total_error = 0
    count = 0

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Calculate the adjusted price with the current beta_candidate
        H_adj_down, H_adj_up = adjusted_barrier(row['T'], row['H'], row['sigma'], m, beta_candidate)
        price_adj = down_and_call_book(S0, K, row['T'], r, q, row['sigma'], row['H'], H_adj_down, H_adj_up)

        # Calculate the percentage error
        error = percentage_error(price_adj, row['price_mc'])
        total_error += error
        count += 1

        # Update current iteration and display progress
        current_iteration += 1
        progress = (current_iteration / total_iterations) * 100
        print(f"Progress: {progress:.2f}%", end='\r')

    # Calculate average error for this beta_candidate
    average_error = total_error / count

    # Check if this is the best beta so far
    if average_error < min_error:
        min_error = average_error
        best_beta = beta_candidate

# Print the best beta and its error
# Best Beta: 0.64, with an average error of: 2.1049123475816645 with all H
# H >= 95 Best Beta: 0.665, with an average error of: 2.236589940094765

print(f"\nBest Beta: {best_beta}, with an average error of: {min_error}")
