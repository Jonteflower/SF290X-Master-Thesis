import pandas as pd
import matplotlib.pyplot as plt

# Load the combined CSV file
df_combined = pd.read_csv('data_up_out.csv')

# Ensure H_percent is calculated and rounded to the nearest whole number
df_combined['H_percent'] = ((df_combined['S0'] - df_combined['H']) / df_combined['S0']) * 100
df_combined['H_percent'] = df_combined['H_percent'].round(0)

# Calculate the error as a percentage between price_iter and price_adj
df_combined['error_percent'] = (abs(df_combined['price_iter'] - df_combined['price_adj']) / df_combined['price_iter']) * 100
df_combined['error_percent'] = df_combined['error_percent'].round(1)  # Keep rounding to one decimal for error

# Group by H_percent and calculate the mean error_percent
avg_error_by_H_percent = df_combined.groupby('H_percent')['error_percent'].mean().reset_index()

# Sort the DataFrame by H_percent to ensure the plot is in descending order
avg_error_by_H_percent = avg_error_by_H_percent.sort_values('H_percent', ascending=False)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(avg_error_by_H_percent['H_percent'], avg_error_by_H_percent['error_percent'], marker='o', linestyle='-')

# Labeling the plot
plt.xlabel('H_percent')
plt.ylabel('Average Error (%)')
plt.title('Average Error vs. H_percent')
plt.grid(True)

# Invert the X-axis so that the largest H_percent starts on the left
plt.gca().invert_xaxis()

# Show the plot
plt.show()
