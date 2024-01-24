import sys
import os
# Adjust the system path to include the parent directory
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

    for T in t_values:
        T_rounded = round(T, 1)
        for H in h_values:
            if type == "custom":
                H_adj_down, H_adj_up = adjusted_barrier_custom(T_rounded, H, S0, K, sigma, m, beta)
            else:
                H_adj_down, H_adj_up = adjusted_barrier(T_rounded, H, sigma, m, beta)

            price_adj = down_and_call_book(S0, K, T_rounded, r, q, sigma, H, H_adj_down, H_adj_up)
            price_mc = df.loc[(df['T'] == T_rounded) & (df['sigma'] == sigma) & (df['H'] == H), 'price_mc'].values[0]
            error = round(abs((price_adj - price_mc) / price_mc * 100), 1)
            errors[H].append(error)

    return errors

def plot_aggregated_errors_across_T(h_values, errors_constant, errors_custom):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate the mean error for each H across all T values for constant beta
    mean_errors_constant = [np.mean(errors_constant[H]) for H in h_values]
    ax.plot(h_values, mean_errors_constant, label=f'Constant Beta = {beta1}')

    # Calculate the mean error for each H across all T values for custom beta
    mean_errors_custom = [np.mean(errors_custom[H]) for H in h_values]
    ax.plot(h_values, mean_errors_custom, label='Custom Beta')

    ax.set_xlabel('H')
    ax.set_ylabel('Mean Percentage Error')
    ax.set_title('Aggregated Mean Error Across All T for Each H')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# Beta values for constant and custom calculations
beta1 = 0.5826  # This is a placeholder, set your own constant beta value
beta_custom = 'custom'  # This is an identifier, not a numerical value

# Data generation for constant and custom betas
errors_constant = generate_data(beta1, "constant")
errors_custom = generate_data(beta1, "custom")  # Assuming the custom function uses the same beta but adjusts it somehow

# Call the new plotting function
plot_aggregated_errors_across_T(h_values, errors_constant, errors_custom)
