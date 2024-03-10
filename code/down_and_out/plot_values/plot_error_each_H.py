import sys
import os

# Add the 'code' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Now you should be able to import from 'equations' and 'generate_data' without issue
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier, adjusted_barrier_custom
from generate_data.base_data import get_base_variables

# The data file contains the exact values
df = pd.read_csv('acc_data.csv')
m = df.iloc[0]['m']
S0 = df.iloc[0]['S0']
K = df.iloc[0]['K']
h_min = df['H'].min()
h_max = df['H'].max()
trading_days = 250
q = 0
r = 0.1

# Iterations for the test case
t_values = np.sort(df['T'].unique().tolist())
h_values = np.sort(df['H'].unique().tolist())

# Sigma values
sigma1 = 0.3
sigma2 = 0.4

def generate_data(beta, sigma, type):
    errors = {H: [] for H in h_values}
    t_vals = []

    for T in t_values:
        T = round(T, 1)
        for H in h_values:
            
            H_adj_down, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)

            price_adj = down_and_call_book(S0, K, T, r, q, sigma, H, H_adj_down, H_adj_up)
            price_mc = df.loc[(df['T'] == T) & (df['sigma'] == sigma) & (df['H'] == H), 'price_iter'].values[0]
            error = round(abs((price_adj - price_mc) / price_mc * 100), 1)

            errors[H].append(error)
        t_vals.append(T)

    return t_vals, errors

def plot_data_per_H(beta1, beta2, sigma1, sigma2, t_vals, errors_beta1, errors_beta2):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    
    for H in h_values:
        axs[0].plot(t_vals, errors_beta1[H], label=f'H = {H}')
        axs[1].plot(t_vals, errors_beta2[H], label=f'H = {H}')

    for ax in axs:
        ax.set_ylabel('Percentage Error')
        ax.legend()
        ax.grid(True)

    axs[0].set_xlabel('T')
    axs[1].set_xlabel('T2')

    axs[0].set_title(f'Error vs T for Beta = {beta1}, Sigma = {sigma1}')
    axs[1].set_title(f'Error vs T for Beta = {beta2}, Sigma = {sigma2}')
    
    plt.tight_layout()
    plt.show()

# Beta values
beta1 = 0.5826
beta2 = 0.5826

# Data generation for both Betas and Sigmas
t_vals, errors_beta1 = generate_data(beta1, sigma1, "regular")
_, errors_beta2 = generate_data(beta2, sigma2, "custom")

# Plot individual H errors for both Betas and Sigmas
plot_data_per_H(beta1, beta2, sigma1, sigma2, t_vals, errors_beta1, errors_beta2)
