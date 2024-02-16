import pandas as pd
import numpy as np

# Load your dataset
df_combined = pd.read_csv('paper_values.csv')

# Add column as a function of the other columns
df_combined['H_log'] = abs(np.log(df_combined['H'] / df_combined['S0']))
df_combined['error_percent'] = abs((df_combined['price_iter'] - df_combined['price_adj']) / df_combined['price_iter']) * 100
df_combined['Product'] = (df_combined['sigma']*np.sqrt(df_combined['T']))

# Add column
df_combined['q'] = 0

df_combined = df_combined.round(8)

# Save the modified DataFrame back to the original CSV file
df_combined.to_csv('paper_values.csv', index=False)