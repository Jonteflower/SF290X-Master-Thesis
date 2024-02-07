import pandas as pd

# Load your dataset
df_combined = pd.read_csv('data_up_out_beta.csv')

# Calculate H_percent and error_percent
df_combined['H_percent'] = ((df_combined['S0'] - df_combined['H']) / df_combined['S0']) * 100
df_combined['error_percent'] = abs((df_combined['price_iter'] - df_combined['price_adj']) / df_combined['price_iter']) * 100

# Now, 'H_percent' and 'error_percent' columns have been appended to the DataFrame.
# You can access and use these columns as needed in 'df_combined'.

# Add the 'q' column to the DataFrame and set it to 0
df_combined['q'] = 0

df_combined = df_combined.round(4)

# Save the modified DataFrame back to the original CSV file
df_combined.to_csv('data_up_out_beta.csv', index=False)
