import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load the CSV file
df_combined = pd.read_csv('data_up_out_beta.csv')  
df_combined = abs(df_combined)

# Define the T and sigma values we are interested in
T_values = [1,1.8, 3, 3.4]
sigma_values = [0.3 ]

# Initialize a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
fig.suptitle('Comparison of Average Optimal Beta vs Percentage Difference')

# Exponential growth function
def exponential_growth(x, a, b):
    return a * np.exp(b * x)

# Power-law relationship
def power_law(x, c, d):
    return c * np.pow(x, d)

# Plot aggregated data for each sigma on the right subplot (ax2)
for sigma in sigma_values:
    # Aggregate data across T for the current sigma
    df_filtered_sigma = df_combined[df_combined['sigma'] == sigma]
    avg_beta_sigma_sorted = df_filtered_sigma.groupby('H_percent')['best_beta'].mean().reset_index()
    
    # Sort the aggregated data in descending order of H_percent before plotting
    #avg_beta_sigma_sorted = avg_beta_sigma.sort_values('H_percent', ascending=False)
    
    # Fit the data to a polynomial for aggregated visualization
    poly_degree_sigma = 3
    poly_coefficients_sigma = np.polyfit(avg_beta_sigma_sorted['H_percent'], avg_beta_sigma_sorted['best_beta'], poly_degree_sigma)
    poly_fit_function_sigma = np.poly1d(poly_coefficients_sigma)
    poly_fit_values_sigma = poly_fit_function_sigma(avg_beta_sigma_sorted['H_percent'])
    
    # Plot the polynomial fit line for the aggregated data
    ax2.plot(avg_beta_sigma_sorted['H_percent'], poly_fit_values_sigma, label=f'Aggregated Sigma={sigma}', linestyle='--')
    ax2.set_xlabel('Percentage Difference (S0 - H) / S0')
    ax2.set_ylabel('Average Optimal Beta')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(max(avg_beta_sigma_sorted['H_percent']), min(avg_beta_sigma_sorted['H_percent']))

# Plot individual data for each T and sigma combination on the left subplot (ax1)
for T in T_values:
    for sigma in sigma_values:
        # Filter the DataFrame for the current T and sigma
        df_filtered = df_combined[(df_combined['T'] == T) & (df_combined['sigma'] == sigma)]
        df_filtered['H_percent'] = ((df_filtered['S0'] - df_filtered['H']) / df_filtered['S0']) * 100
        avg_beta = df_filtered.groupby('H_percent')['best_beta'].mean().reset_index()
        
        # Sort each individual data set in descending order of H_percent before plotting
        avg_beta_sorted = avg_beta.sort_values('H_percent', ascending=False)
        
        # Fit the data to a polynomial
        poly_degree = 3
        poly_coefficients = np.polyfit(avg_beta_sorted['H_percent'], avg_beta_sorted['best_beta'], poly_degree)
        poly_fit_function = np.poly1d(poly_coefficients)
        poly_fit_values = poly_fit_function(avg_beta_sorted['H_percent'])
        
        # Plot the polynomial fit line for each combination
        ax1.plot(avg_beta_sorted['H_percent'], poly_fit_values, label=f'T={T} Sigma={sigma}, σ*sqrt(T)={round(sigma*np.sqrt(T),2)} ', linestyle='--')
        ax1.set_xlabel('Percentage Difference (S0 - H) / S0')
        ax1.set_ylabel('Average Optimal Beta')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim(max(avg_beta_sorted['H_percent']), min(avg_beta_sorted['H_percent']))

plt.tight_layout()
plt.show()
