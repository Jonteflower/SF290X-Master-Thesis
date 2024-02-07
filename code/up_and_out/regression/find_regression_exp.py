import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load the CSV file
df_combined = pd.read_csv('paper_values.csv')

# Define the exponential model for curve fitting
def exponential_model(X, a, b1, b2):
    h_percent, sigma_sqrt_T = X
    return a * np.exp(b1 * h_percent + b2 * sigma_sqrt_T)

# Check if the array is strictly increasing
def is_increasing(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

# Find the index where Beta starts strictly increasing
def find_strictly_increase_index(data, T_val, sigma_val, threshold):
    filtered_data = data[(data['sigma'] == sigma_val) & (data['T'] == T_val)]
    sorted_data = filtered_data.sort_values(by='H_percent', ascending=True)
    
    if filtered_data.empty:
        return None
    
    for i in range(len(sorted_data)):
        if sorted_data.iloc[i]['best_beta'] > threshold:
            return i
    return None

# Define the T and sigma values we are interested in
T_values = [1, 2, 5]
sigma_values = [0.2, 0.3, 0.5]
beta_threshold = 0.5826  # The Beta value threshold

# Adjust the dataset
adjusted_data = pd.DataFrame()
for T in T_values:
    for sigma in sigma_values:
        index_to_start_increasing = find_strictly_increase_index(df_combined, T, sigma, beta_threshold)
        if index_to_start_increasing is not None:
            strictly_increasing_data = df_combined[(df_combined['sigma'] == sigma) & (df_combined['T'] == T)].iloc[index_to_start_increasing:]
            adjusted_data = pd.concat([adjusted_data, strictly_increasing_data], ignore_index=True)

# Prepare the independent variables 'H_percent' and 'sigma_sqrt_T' and the dependent variable 'Beta'
adjusted_data['sigma_sqrt_T'] = adjusted_data['sigma'] * np.sqrt(adjusted_data['T'])
adjusted_data['H_percent'] = ((adjusted_data['S0'] - adjusted_data['H']) / adjusted_data['S0']) * 100
adjusted_data['Beta'] = adjusted_data['best_beta']  # Assuming 'best_beta' is the column for Beta

# Prepare the data for fitting
X = (adjusted_data['H_percent'].values, adjusted_data['sigma_sqrt_T'].values)
y = adjusted_data['Beta'].values

# Fit the exponential model
params, covariance = curve_fit(exponential_model, X, y)

# Extract the parameters
a, b1, b2 = params

# Print the equation of the fitted exponential model
print(f"Fitted exponential equation: Beta = {a:.4f} * exp({b1:.4f} * H_percent + {b2:.4f} * sigma_sqrt_T)")

# Print the fitted parameters
print(f"Fitted parameters: a = {a:.4f}, b1 = {b1:.4f}, b2 = {b2:.4f}")