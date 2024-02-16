import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the combined CSV file
df_combined = pd.read_csv('paper_values.csv')

# Filter the DataFrame for T=2
df_filtered = df_combined[df_combined['T'] == 2]

# Calculate the percentage difference
#df_filtered['H_log'] = ((df_filtered['S0'] - df_filtered['H']) / df_filtered['S0']) * 100
df_filtered['H_log'] = abs(np.log(df_filtered['H']/df_filtered['S0']) )

# Calculate the error between price_iter and price_adj
df_filtered['error'] = (abs(df_filtered['price_iter'] - df_filtered['price_adj'])/df_filtered['price_iter'])*100

# Create a figure
plt.figure(figsize=(10, 6))

# Plot settings
colors = ['black', 'blue', 'red', 'green']
sigma_values = [0.3, 0.5]

# Plot the error for each K value and each sigma value
for i, K in enumerate(df_filtered['K'].unique()):
    for sigma in sigma_values:
        df_plot = df_filtered[(df_filtered['K'] == K) & (df_filtered['sigma'] == sigma)]
        if not df_plot.empty:
            # Ensure H_log is sorted in descending order for plotting
            df_plot = df_plot.sort_values('H_log', ascending=False)
            plt.plot(df_plot['H_log'], df_plot['error'], 
                     label=f'Error S0={K} Sigma={sigma}', color=colors[i % len(colors)], linestyle='--' if sigma == 0.5 else '-')

plt.xlabel('Percentage Difference (S0 - H) / S0)')
plt.ylabel('Error')
plt.title('Error vs percentage of H/K')
plt.legend()
plt.grid(True)

# Invert the X-axis so that 0% is on the right
#plt.xlim(max(df_filtered['H_log']), 0)
plt.xlim(max(df_filtered['H_log']), 0)


pd.options.mode.chained_assignment = None  # default='warn'