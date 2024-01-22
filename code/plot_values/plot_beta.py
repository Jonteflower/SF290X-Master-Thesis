import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from generate_data.data import get_beta_values

def plot_beta_values(beta_values):
    # Extract keys (H values) and values (beta values)
    H_values = list(beta_values.keys())
    beta_values = list(beta_values.values())

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(H_values, beta_values, color='skyblue')
    plt.xlabel('H Values')
    plt.ylabel('Optimal Beta Values')
    plt.title('Optimal Beta Values for Different H Values')
    plt.xticks(H_values)  # Set H values as x-axis ticks
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Get the beta values from the function
beta_values = get_beta_values()

# Plot the beta values
plot_beta_values(beta_values)
