import pandas as pd
import numpy as np

def find_strict_increase_start(df):
    # Ensure sorting by 'H_percent_rounded' if that's the column you're using
    df = df.sort_values('H_percent_rounded', ascending=False).reset_index(drop=True)
    increasing = False
    start_point = None

    for index in range(1, len(df)):
        if df.loc[index, 'error_percent_rounded'] > df.loc[index - 1, 'error_percent_rounded']:
            if not increasing:  # Found the start of an increase
                start_point = df.loc[index, 'H_percent_rounded']
                increasing = True
        else:
            # If the trend does not continue strictly, reset the increasing flag
            increasing = False

    # Return the start point if an increase was found, otherwise return None
    return start_point if increasing else None

# Load the combined CSV file
df_combined = pd.read_csv('data.csv')

# Filter the DataFrame for T=2
df_filtered = df_combined[df_combined['T'] == 0.2].copy()

# Calculate the percentage difference and error as a percentage
df_filtered['H_percent'] = ((df_filtered['S0'] - df_filtered['H']) / df_filtered['S0']) * 100
df_filtered['error_percent'] = (abs(df_filtered['price_iter'] - df_filtered['price_adj']) / df_filtered['price_iter']) * 100

# Round H_percent for aggregation and error_percent for display
df_filtered['H_percent_rounded'] = df_filtered['H_percent'].round(0)
df_filtered['error_percent_rounded'] = df_filtered['error_percent'].round(2)

# Separate the dataframes for each sigma value and group by H_percent_rounded
sigma_values = [0.3, 0.5]
for sigma in sigma_values:
    df_sigma = df_filtered[df_filtered['sigma'] == sigma]
    avg_error_by_H_percent = df_sigma.groupby('H_percent_rounded')['error_percent_rounded'].mean().reset_index()

    # Find the start of strict increase
    increase_start = find_strict_increase_start(avg_error_by_H_percent)
    print(f"Strict increase starts at H_percent {increase_start} for sigma {sigma}")
