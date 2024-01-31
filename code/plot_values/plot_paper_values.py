import pandas as pd
import matplotlib.pyplot as plt

# Load the combined CSV file
df_combined = pd.read_csv('paper_values.csv')

# Filter the DataFrame for T=2
df_filtered = df_combined[df_combined['T'] == 2]

# Calculate the percentage difference
df_filtered['H_percent'] = ((df_filtered['S0'] - df_filtered['H']) / df_filtered['S0']) * 100

# Create a figure
plt.figure(figsize=(10, 6))

# Plot settings
colors = ['black', 'blue', 'red', 'green']
sigma_values = [0.3, 0.5]

# Plot optimal Beta for each K value and each sigma value
for i, K in enumerate(df_filtered['K'].unique()):
    for sigma in sigma_values:
        df_plot = df_filtered[(df_filtered['K'] == K) & (df_filtered['sigma'] == sigma)]
        if not df_plot.empty:
            plt.plot(df_plot['H_percent'], df_plot['best_beta'], 
                     label=f'Optimal Beta K={K} Sigma={sigma}', color=colors[i % len(colors)], linestyle='--' if sigma == 0.5 else '-')

# Horizontal line for Beta = 0.5826
plt.axhline(y=0.5826, color='gray', linestyle='--', label='Beta = 0.5826')

plt.xlabel('Percentage Difference (S0 - H) / S0')
plt.ylabel('Optimal Beta')
plt.title('Optimal Beta vs Percentage Difference for Different K and Sigma Values at T=2')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('beta_plots_T2_sigma.png')

# Show the plot
plt.show()
