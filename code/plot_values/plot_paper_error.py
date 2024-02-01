import pandas as pd
import matplotlib.pyplot as plt

# Load the combined CSV file
df_combined = pd.read_csv('paper_values.csv')

# Filter the DataFrame for T=2
df_filtered = df_combined[df_combined['T'] == 2]

# Calculate the percentage difference
df_filtered['H_percent'] = ((df_filtered['S0'] - df_filtered['H']) / df_filtered['S0']) * 100

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
            # Ensure H_percent is sorted in descending order for plotting
            df_plot = df_plot.sort_values('H_percent', ascending=False)
            plt.plot(df_plot['H_percent'], df_plot['error'], 
                     label=f'Error K={K} Sigma={sigma}', color=colors[i % len(colors)], linestyle='--' if sigma == 0.5 else '-')

plt.xlabel('Percentage Difference (S0 - H) / S0)')
plt.ylabel('Error')
plt.title('Error vs percentage of H/K')
plt.legend()
plt.grid(True)

# Invert the X-axis so that 0% is on the right
plt.xlim(max(df_filtered['H_percent']), 0)


pd.options.mode.chained_assignment = None  # default='warn'

# Filter the DataFrame for T=2
df_filtered = df_combined[df_combined['T'] == 2]

# Calculate the percentage difference
df_filtered['H_percent'] = ((df_filtered['S0'] - df_filtered['H']) / df_filtered['S0']) * 100

# Calculate the error as a percentage between price_iter and price_adj
df_filtered['error_percent'] = (abs(df_filtered['price_iter'] - df_filtered['price_adj']) / df_filtered['price_iter']) * 100

df_filtered['H_percent'] = df_filtered['H_percent'].round(0)  # Round to nearest whole number for aggregation
df_filtered['error_percent'] = df_filtered['error_percent'].round(1)  # Keep rounding to one decimal for error

# Filter for H_percent between 5% and 10%
df_filtered = df_filtered[(df_filtered['H_percent'] >= 2) & (df_filtered['H_percent'] <= 20)]

# Separate the dataframes for each sigma value
df_sigma_03 = df_filtered[df_filtered['sigma'] == 0.3]
df_sigma_05 = df_filtered[df_filtered['sigma'] == 0.5]

# Group by H_percent and calculate the mean error_percent for sigma 0.3
avg_error_by_H_percent_sigma_03 = df_sigma_03.groupby('H_percent')['error_percent'].mean().reset_index()

# Group by H_percent and calculate the mean error_percent for sigma 0.5
avg_error_by_H_percent_sigma_05 = df_sigma_05.groupby('H_percent')['error_percent'].mean().reset_index()

# Printing the average error for each percentage increase in H for both sigma values
print("Average error for sigma 0.3:")
print(avg_error_by_H_percent_sigma_03)
print("\nAverage error for sigma 0.5:")
print(avg_error_by_H_percent_sigma_05)

# Show the plot
plt.show()
