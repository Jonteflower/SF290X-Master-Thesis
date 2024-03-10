import sys
import os

# Add the 'code' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sample DataFrame
file = 'acc_data_m.csv'
df = pd.read_csv(file)

# Filter out best_beta < 0.55
df_filtered = df[(df['best_beta'] >= 0.57) & (df['m'] == 25)]

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

# Filter out different versions using variables
df_t1_sigma1 = df_filtered[(df_filtered['T'] == t1) & (df_filtered['sigma'] == sigma1)]
df_t2_sigma2 = df_filtered[(df_filtered['T'] == t2) & (df_filtered['sigma'] == sigma2)]
df_t3_sigma3 = df_filtered[(df_filtered['T'] == t3) & (df_filtered['sigma'] == sigma3)]

# Define product values for each version
prod_t1_sigma1 = np.round(np.sqrt(t1) * sigma1, 2)
prod_t2_sigma2 = np.round(np.sqrt(t2) * sigma2, 2)
prod_t3_sigma3 = np.round(np.sqrt(t3) * sigma3, 2)

# Create figure and axes for subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plotting histogram with filtered versions and different colors
axs[0, 0].hist(df_t1_sigma1['best_beta'], bins=20, edgecolor='black', alpha=0.5, color='blue', label=f'T={t1}, sigma={sigma1}, prod={prod_t1_sigma1}')
axs[0, 0].hist(df_t2_sigma2['best_beta'], bins=20, edgecolor='black', alpha=0.5, color='green', label=f'T={t2}, sigma={sigma2}, prod={prod_t2_sigma2}')
axs[0, 0].hist(df_t3_sigma3['best_beta'], bins=20, edgecolor='black', alpha=0.5, color='red', label=f'T={t3}, sigma={sigma3}, prod={prod_t3_sigma3}')

# Adding legend to the first plot
axs[0, 0].legend()
axs[0, 0].set_xlabel('best_beta')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].set_title('Histogram of best_beta for Different Versions')

# Plotting scatter plot with filtered versions sorted by x_axis_key
df_t1_sigma1_sorted = df_t1_sigma1.sort_values(by=x_axis_key)
df_t2_sigma2_sorted = df_t2_sigma2.sort_values(by=x_axis_key)
df_t3_sigma3_sorted = df_t3_sigma3.sort_values(by=x_axis_key)

# Plotting scatter plot with filtered versions
scatter_ax = axs[0, 1]
scatter_ax.scatter(df_t1_sigma1_sorted[x_axis_key], df_t1_sigma1_sorted['best_beta'], color='blue', label=f'T={t1}, sigma={sigma1}, prod={prod_t1_sigma1}')
scatter_ax.scatter(df_t2_sigma2_sorted[x_axis_key], df_t2_sigma2_sorted['best_beta'], color='green', label=f'T={t2}, sigma={sigma2}, prod={prod_t2_sigma2}')
scatter_ax.scatter(df_t3_sigma3_sorted[x_axis_key], df_t3_sigma3_sorted['best_beta'], color='red', label=f'T={t3}, sigma={sigma3}, prod={prod_t3_sigma3}')

# Adding legend to the second plot
scatter_ax.legend()
scatter_ax.set_ylabel('best_beta')
scatter_ax.set_title('Scatter Plot of best_beta for Different Versions')

# Reverse the x-axis for the scatter plot if x_axis_key is not 'H'
if x_axis_key != 'H':
    scatter_ax.invert_xaxis()

# Plotting lines of beta
axs[1, 0].plot(df_t1_sigma1_sorted[x_axis_key], df_t1_sigma1_sorted['best_beta'], color='blue', label=f'T={t1}, sigma={sigma1}, prod={prod_t1_sigma1}')
axs[1, 0].plot(df_t2_sigma2_sorted[x_axis_key], df_t2_sigma2_sorted['best_beta'], color='green', label=f'T={t2}, sigma={sigma2}, prod={prod_t2_sigma2}')
axs[1, 0].plot(df_t3_sigma3_sorted[x_axis_key], df_t3_sigma3_sorted['best_beta'], color='red', label=f'T={t3}, sigma={sigma3}, prod={prod_t3_sigma3}')

# Adding legend to the third plot
axs[1, 0].legend()
axs[1, 0].set_xlabel(x_axis_key)
axs[1, 0].set_ylabel('best_beta')
axs[1, 0].set_title('Lines of best_beta for Different Versions')

# Generate Q-Q plot with a logistic distribution
stats.probplot(df_filtered['best_beta'], dist="logistic", plot=axs[1, 1])

# Adding title to the Q-Q plot
axs[1, 1].set_title('Q-Q Plot with Logistic Distribution')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
