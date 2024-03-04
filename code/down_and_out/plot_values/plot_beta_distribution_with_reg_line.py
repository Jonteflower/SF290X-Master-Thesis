import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add the 'code' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sample DataFrame
file = 'acc_data.csv'
df = pd.read_csv(file)

# Filter out best_beta < 0.55
df_filtered = df[df['best_beta'] >= 0.55]
x_axis_key = 'H'

# Randomly select a combination from df_filtered
sample1 = df_filtered.sample(1).iloc[0]
sigma1 = sample1['sigma']
t1 = sample1['T']

sample2 = df_filtered.sample(1).iloc[0]
sigma2 = sample2['sigma']
t2 = sample2['T']

sample3 = df_filtered.sample(1).iloc[0]
sigma3 = sample3['sigma']
t3 = sample3['T']

##### Regression line for Beta
"""
def regression_beta_engineer(T, sigma, H, S0):
    beta_start = 0.5826
    beta_end = 0.72
    Sigma_sqrt_T = sigma * np.sqrt(T)
    H_log_start = 0.1571 * Sigma_sqrt_T
    H_start_increase = round(S0 * np.exp(-H_log_start))  # Determine the start of increase
    
    # End of the interval
    H_end = S0 - 1
    
    # Log condition to switch equations
    H_log = abs(np.log(H/S0))
    
    # If the condition is met, use the exponential growth equation
    if H_log < H_log_start:
        # Calculate the growth rate 'b'
        b = (beta_end / beta_start)**(1 / ((H_end - H_start_increase))) 
        a = beta_start/(b**(H_start_increase))
        
        # Calculate the exponential increase
        beta = a*b**H
        print("H_start_increase ",H_start_increase, "a ",a, "b ", b, "beta ", beta )
    else:
        # If the condition is not met, beta remains at beta_start
        beta = beta_start
    
    return beta
"""

def regression_beta_engineer(T, sigma, H, S0):
    beta_start = 0.5826
    Sigma_sqrt_T = sigma * np.sqrt(T)

    #### TODO find this as a function of 
    #beta_end = 0.7174 ### Make this a function of sigma*sqrt(T) maybe over m
    beta_end = 0.71432 / (1 + np.exp(-4.5498*(Sigma_sqrt_T + 0.45534)))
     
    ### This is from the reg of increase point both Logistic and quadratic can be used
    #H_log_start =  -(7.3896e-02)*Sigma_sqrt_T**2 + (2.2475e-01)*Sigma_sqrt_T + -5.4974e-03
    H_log_start = 0.15609 / (1 + np.exp(-4.5799*(Sigma_sqrt_T - 0.4876)))
    
    H_start_increase = round(S0 * np.exp(-H_log_start))  # Determine the start of increase
    H_end = S0 - 1

    # Start with a constant beta
    beta = beta_start

    # If we've reached the start of the increase, switch to the polynomial curve
    if H >= H_start_increase:
        # These are placeholders and should be solved based on your specific conditions
        a = (beta_end - beta_start) / ((H_end - H_start_increase)**2)
        b = -2 * a * H_start_increase
        c = beta_start + a * H_start_increase**2
        
        # Polynomial growth formula
        beta = a * H**2 + b * H + c

    return beta

# Define the combinations for T and sigma
combinations = [(t1, sigma1), (t2, sigma2), (t3, sigma3)]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines for actual best_beta values and estimated best fit
for T, sigma in combinations:
    # Filter the DataFrame for each combination
    subset = df[(df['T'] == T) & (df['sigma'] == sigma)]
    
    # Sort by H for consistent line plotting
    subset_sorted = subset.sort_values(by='H')
    
    # Plot the actual best_beta values
    plt.plot(subset_sorted['H'], subset_sorted['best_beta'], label=f'T={T}, sigma={sigma}')

    # Calculate the estimated best fit betas
    subset_sorted['estimated_beta'] = subset_sorted.apply(lambda row: regression_beta_engineer(row['T'], row['sigma'], row['H'], row['S0']), axis=1)
    
    # Plot the estimated best fit line
    plt.plot(subset_sorted['H'], subset_sorted['estimated_beta'], '--', label=f'Est. Beta T={T}, sigma={sigma}')

# Add labels and title
plt.xlabel('H')
plt.ylabel('best_beta')
plt.title('Actual Best Beta vs general Equation for Best Beta')
plt.legend()

# Show the plot
plt.show()