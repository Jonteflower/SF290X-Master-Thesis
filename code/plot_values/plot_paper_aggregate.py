import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df_combined = pd.read_csv('data_with_beta.csv')

# Define the T and sigma values we are interested in
T_values = [2, 5]
sigma_values = [0.3, 0.5]

# Initialize a figure
plt.figure(figsize=(10, 6))

# For each T and sigma value, calculate the mean of best_beta and plot
for T in T_values:
    T = round(T, 1)
    for sigma in sigma_values:
        # Filter the DataFrame for the current T and sigma
        df_filtered = df_combined[(df_combined['T'] == T) & (df_combined['sigma'] == sigma)].copy()
        
        # Check if df_filtered is empty
        if df_filtered.empty:
            print(f"Missing value for T={T}, Sigma={sigma}")
            continue  # Skip the current iteration if there are no data points
        
        # Calculate H_percent
        df_filtered['H_percent'] = ((df_filtered['S0'] - df_filtered['H']) / df_filtered['S0']) * 100
        
        # Group by H_percent and calculate the mean of best_beta
        avg_beta = df_filtered.groupby('H_percent')['best_beta'].mean().reset_index()
        
        # Plot the non exponential lines for this 
        #plt.plot(avg_beta['H_percent'], avg_beta['best_beta'], label=f'Average Optimal Beta T={T:.1f} Sigma={sigma}')
        
        # Fit the data to a polynomial
        poly_degree = 3  # Degree of polynomial to fit
        poly_coefficients = np.polyfit(avg_beta['H_percent'], avg_beta['best_beta'], poly_degree)
        poly_fit_function = np.poly1d(poly_coefficients)
        
        # Evaluate the polynomial fit function at the H_percent points
        poly_fit_values = poly_fit_function(avg_beta['H_percent'])
        
        # Plot the polynomial fit line
        plt.plot(avg_beta['H_percent'], poly_fit_values, label=f'Polyfit T={T:.1f} Sigma={sigma} Sum={round(np.sqrt(T)*sigma, 2)}', linestyle='--')
        
# Horizontal line for Beta = 0.5826
plt.axhline(y=0.5826, color='gray', linestyle='--', label='Baseline Beta')

# Labels and title
plt.xlabel('Percentage Difference (S0 - H) / S0')
plt.ylabel('Average Optimal Beta')
plt.title('Average Optimal Beta vs Percentage Difference for Various T and Sigma Values')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('average_beta_plots_combined.png')

# Show the plot
plt.show()
