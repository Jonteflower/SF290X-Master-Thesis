import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the DataFrame
file = 'acc_data_m.csv'
df = pd.read_csv(file)
df_filtered = df[(df['best_beta'] >= 0.55)]


# Function to filter and return samples based on T, sigma, and m values
def filter_samples(df, t, sigma, m):
    return df[(df['T'] == t) & (df['sigma'] == sigma) & (df['m'] == m)]

# Select a random sample of unique combinations of T and sigma for plotting
num_combinations_to_plot = 3  # You can adjust this number as needed
unique_combinations = df[['T', 'sigma']].drop_duplicates().sample(n=num_combinations_to_plot)

# Create figure for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Colors for each combination
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
color_index = 0

for _, row in unique_combinations.iterrows():
    T, sigma = row['T'], row['sigma']
    
    # Filter for M=25 and M=50
    df_m25 = filter_samples(df, T, sigma, 25)
    df_m50 = filter_samples(df, T, sigma, 50)
    
    if df_m25.empty or df_m50.empty:
        continue  # Skip if no data for the combination
    
    # Sort values for plotting
    df_m25_sorted = df_m25.sort_values(by='H')
    df_m50_sorted = df_m50.sort_values(by='H')
    
    # Plot M=25 as solid, M=50 as dotted with the same color
    ax.plot(df_m25_sorted['H'], df_m25_sorted['best_beta'], label=f'T={T}, sigma={sigma}, M=25, prod={round(sigma*np.sqrt(T/25),4)}', color=colors[color_index], linestyle='-')
    ax.plot(df_m50_sorted['H'], df_m50_sorted['best_beta'], label=f'T={T}, sigma={sigma}, M=50, prod={round(sigma*np.sqrt(T/50),4)}', color=colors[color_index], linestyle=':')
    
    color_index = (color_index + 1) % len(colors)  # Cycle through colors

# Add legend, labels, and title
ax.legend()
ax.set_xlabel('H')
ax.set_ylabel('best_beta')
ax.set_title('Random Comparison of best_beta for Different T, sigma, and M values')

# Show plot
plt.show()
