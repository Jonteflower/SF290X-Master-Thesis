
import sys
import os
# Adjust the system path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from generate_data.find_beta import find_optimal_beta

# Assuming find_optimal_beta function signature looks something like this:
# find_optimal_beta(S0, K, r, q, sigma, m, H, T, price)

# Load the data
df = pd.read_csv('data.csv')

# Define a function to apply to each row
def apply_find_beta(row):
    # Assuming q is not provided in the dataset and setting it to 0 or any applicable value
    q = 0
    # Call the find_optimal_beta function with parameters from the row
    best_beta = find_optimal_beta(row['S0'], row['K'], row['r'], q, row['sigma'], row['m'], row['H'], row['T'], row['price_mc'])
    return best_beta[0]

# Apply the function to each row and create a new column 'best_beta'
df['best_beta'] = df.apply(apply_find_beta, axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv('data_with_beta.csv', index=False)
