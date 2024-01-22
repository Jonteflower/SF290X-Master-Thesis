import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data.data import get_beta_values
import matplotlib.pyplot as plt

def plot_beta_values(beta_values):
    # Extract keys (H values) and values (beta values)
    H_values = list(beta_values.keys())
    beta_values = list(beta_values.values())

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(H_values, beta_values, linestyle='-')
    plt.xlabel('H Values')
    plt.ylabel('Optimal Beta Values')
    plt.title('Optimal Beta Values for Different H Values')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Get the beta values from the function
beta_values = get_beta_values()

# Plot the beta values as a line graph
plot_beta_values(beta_values)