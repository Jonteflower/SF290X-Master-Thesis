import sys
import os
# Adjust the system path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from generate_data.data import get_beta_values

# Get the beta values for different sigma values
beta_values = get_beta_values()

# Initialize subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Iterate over each sigma's beta values for line plot
for sigma, sigma_beta_values in beta_values.items():
    H_values = np.array(list(sigma_beta_values.keys()))
    optimal_betas = np.array(list(sigma_beta_values.values()))

    axs[0].plot(H_values, optimal_betas, label=f'Sigma={sigma}')

# Iterate over each sigma's beta values for scatter plot with polynomial fit
for sigma, sigma_beta_values in beta_values.items():
    H_values = np.array(list(sigma_beta_values.keys()))
    optimal_betas = np.array(list(sigma_beta_values.values()))

    # Perform polynomial fitting with a degree of 2 (quadratic)
    coefficients = np.polyfit(H_values, optimal_betas, 2)
    p = np.poly1d(coefficients)

    # Generate a range of H values for plotting the fit
    H_fit = np.linspace(min(H_values), max(H_values), 100)
    beta_fit = p(H_fit)

    axs[1].scatter(H_values, optimal_betas, label=f'Sigma={sigma} Data')
    axs[1].plot(H_fit, beta_fit, label=f'Sigma={sigma} Fit: {p}')

# Setting labels, titles, and legends
axs[0].set_xlabel('H Values')
axs[0].set_ylabel('Optimal Beta Values')
axs[0].set_title('Line Plot of Beta Values for Different Sigmas')
axs[0].legend()
axs[0].grid(True)

axs[1].set_xlabel('H Values')
axs[1].set_ylabel('Optimal Beta Values')
axs[1].set_title('Scatter Plot with Polynomial Fit for Different Sigmas')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
