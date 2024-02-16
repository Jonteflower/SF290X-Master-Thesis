import pandas as pd

df = pd.read_csv('paper_values.csv')
# Filter the DataFrame for T=2
df_filtered = df[df['T'] == 2]

df_filtered['H_log'] = df_filtered['H_log'].round(4)  # Round to nearest whole number for aggregation
df_filtered['error_percent'] = df_filtered['error_percent'].round(1)  # Keep rounding to one decimal for error

# Filter for H_log between 5% and 10%
#df_filtered = df_filtered[(df_filtered['H_log'] >= 2) & (df_filtered['H_log'] <= 20)]

# Separate the dataframes for each sigma value
df_sigma_03 = df_filtered[df_filtered['sigma'] == 0.3]
df_sigma_05 = df_filtered[df_filtered['sigma'] == 0.4]

# Group by H_log and calculate the mean error_percent for sigma 0.3
avg_error_by_H_log_sigma_03 = df_sigma_03.groupby('H_log')['error_percent'].mean().reset_index()

# Group by H_log and calculate the mean error_percent for sigma 0.5
avg_error_by_H_log_sigma_05 = df_sigma_05.groupby('H_log')['error_percent'].mean().reset_index()

# Printing the average error for each percentage increase in H for both sigma values
print("Average error for sigma 0.3:")
print(avg_error_by_H_log_sigma_03)
print("\nAverage error for sigma 0.5:")
print(avg_error_by_H_log_sigma_05)

