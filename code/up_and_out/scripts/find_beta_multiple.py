import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import os
import sys
from equations.adjusted_barrier import adjusted_barrier
from generate_data.find_beta import find_optimal_beta

# Assuming the CSV file already exists and contains all necessary columns, including 'price_iter'
csv_file_path = 'data_up_out.csv'  # Update this to the path of your existing CSV file

# Load the existing CSV data into a DataFrame
df = pd.read_csv(csv_file_path)

# You might need to adjust the column names according to your CSV
def compute_best_beta(row):
    S0, K, r, m, T, H, sigma, trading_days, price_iter, q, n = row['S0'], row['K'], row['r'], row['m'], row['T'], row['H'], row['sigma'], row['trading_days'], row['price_iter'], 0, 1*10**7
    # Assuming price_iter is the first element if it's stored as a tuple in the CSV
    if isinstance(price_iter, str):
        price_iter = eval(price_iter)[0]
    _, H_adj_up = adjusted_barrier(T, H, sigma, m, beta=0.5826)  # Assuming a default beta value to get H_adj_up
    best_beta_value, _ = find_optimal_beta(S0, K, r, q, sigma, m, H, T, price_iter)
    print(best_beta_value)
    return best_beta_value

# Apply the function to each row to compute the best beta
# Make sure beta is not already a column; if it is, adjust the column name accordingly
df['best_beta'] = df.apply(compute_best_beta, axis=1)

# Save the updated DataFrame to a new CSV file
new_csv_file_path = 'updated_data_with_best_beta.csv'  # Update this to your desired new CSV file path
df.to_csv(new_csv_file_path, index=False)

print(f"Updated data saved to '{new_csv_file_path}'.")
