import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier, adjusted_barrier_custom
from generate_data.data import get_base_variables

# Base variables
m, r, T, sigma, S0, K, trading_days, beta_default, H_init, q = get_base_variables()

# Iterations for the test case
t_values = np.arange(0.2, 5.1, 0.1)
h_values = range(90, 100)

# The data file contains the exact values
df = pd.read_csv('data.csv')

def generate_data(beta, type):
    errors = {H: [] for H in h_values}
    t_vals = []

    for T in t_values:
        T = round(T, 1)
        for H in h_values:
            
            if type == "custom":
                H_adj_down, H_adj_up = adjusted_barrier_custom(T, H, S0, K, sigma, m, beta)
            else:
                H_adj_down, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)

            price_adj = down_and_call_book(S0, K, T, r, q, sigma, H, H_adj_down, H_adj_up)
            price_mc = df.loc[(df['T'] == T) & (df['sigma'] == sigma) & (df['H'] == H), 'price_mc'].values[0]
            error = round(abs((price_adj - price_mc) / price_mc * 100), 1)

            errors[H].append(error)
        t_vals.append(T)

    return t_vals, errors

def plot_aggregated_data(beta1, beta2, beta3, t_vals, errors_beta1, errors_beta2, errors_beta3):
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_errors_beta1 = [np.mean([errors_beta1[H][i] for H in h_values]) for i in range(len(t_vals))]
    mean_errors_beta2 = [np.mean([errors_beta2[H][i] for H in h_values]) for i in range(len(t_vals))]
    mean_errors_beta3 = [np.mean([errors_beta3[H][i] for H in h_values]) for i in range(len(t_vals))]

    ax.plot(t_vals, mean_errors_beta1, label=f'Beta = {beta1}')
    ax.plot(t_vals, mean_errors_beta2, label=f'Beta = {beta2}')
    ax.plot(t_vals, mean_errors_beta3, label=f'Custom Beta = {beta3}')
    ax.set_xlabel('T')
    ax.set_ylabel('Mean Percentage Error')
    ax.set_title('Mean Error vs T for Different Betas')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# Beta values
beta1 = 0.5826
beta2 = 0.64
beta3 = 0.5826  

# Data generation for all Betas
t_vals, errors_beta1 = generate_data(beta1, "constant")
_, errors_beta2 = generate_data(beta2, "constant")
_, errors_custom_beta = generate_data(beta3, "custom")  # Generate data for custom beta

# Plot aggregated error for all Betas
plot_aggregated_data(beta1, beta2, beta3, t_vals, errors_beta1, errors_beta2, errors_custom_beta)
