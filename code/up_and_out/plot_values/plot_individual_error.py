import sys
import os

# Add the 'code' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Now you should be able to import from 'equations' and 'generate_data' without issue
from equations.up_and_out_call import up_and_out_call
from equations.adjusted_barrier import adjusted_barrier, adjusted_barrier_custom
from generate_data.base_data import get_base_variables

# Base variables
m, r, T, sigma, S0, K, trading_days, beta_default, H_init, q = get_base_variables()

# Iterations for the test case
t_values = np.arange(0.2, 5.3, 0.4)
h_values = range(115, 155)

# The data file contains the exact values
df = pd.read_csv('data_up_out.csv')

def generate_data(beta, type):
    errors = {H: [] for H in h_values}
    t_vals = []

    for T in t_values:
        T = round(T, 1)
        for H in h_values:
            
            if type == "custom":
                _, H_adj_up = adjusted_barrier_custom(T, H, S0, K, sigma, m, beta, df)
            else:
                _, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)

            price_adj = up_and_out_call(S0, K, T, r, q, sigma, H_adj_up, )
            price_mc = df.loc[(df['T'] == T) & (df['sigma'] == sigma) & (df['H'] == H), 'price_iter'].values[0]
            error = round(abs((price_adj - price_mc) / price_mc * 100), 1)

            errors[H].append(error)
        t_vals.append(T)

    return t_vals, errors

def plot_data_per_H(beta1, beta2, t_vals, errors_beta1, errors_beta2):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    
    for H in h_values:
        axs[0].plot(t_vals, errors_beta1[H], label=f'H = {H}')
        axs[1].plot(t_vals, errors_beta2[H], label=f'H = {H}')

    for ax in axs:
        ax.set_xlabel('T')
        ax.set_ylabel('Percentage Error')
        ax.legend()
        ax.grid(True)

    axs[0].set_title(f'Error vs T for Beta = {beta1}')
    axs[1].set_title(f'Error vs T for Beta = {beta2}')
    
    plt.tight_layout()
    plt.show()

# Beta values
beta1 = 0.5826
beta2 = 0.5826

# Data generation for both Betas
t_vals, errors_beta1 = generate_data(beta1, "regular")
_, errors_beta2 = generate_data(beta2, "custom")

# Plot individual H errors for both Betas
plot_data_per_H(beta1, beta2, t_vals, errors_beta1, errors_beta2)