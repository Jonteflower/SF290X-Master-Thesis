import pandas as pd
import numpy as np

# Load your dataset
df_combined = pd.read_csv('paper_values.csv')
df_combined['m'] = 50
df_combined['q'] = 0

# Add column as a function of the other columns
df_combined['H_log'] = abs(np.log(df_combined['H'] / df_combined['S0']))
df_combined['error_percent'] = abs((df_combined['price_iter'] - df_combined['price_adj']) / df_combined['price_iter']) * 100
df_combined['Product'] = (df_combined['sigma']*np.sqrt(df_combined['T']))

df_combined = df_combined.round(8)

# Columns to drop
columns_to_drop = ['price_adj_custom', 'error', 'error_custom', 'error_percent']

# Drop the specified columns
df_dropped = df_combined.drop(columns=columns_to_drop, errors='ignore')

# Save the modified DataFrame back to the original CSV file
df_dropped.to_csv('paper_values.csv', index=False)