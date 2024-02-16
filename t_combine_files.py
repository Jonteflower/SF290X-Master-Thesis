import pandas as pd

# Load the datasets
df1 = pd.read_csv('paper_values_300.csv')
df2 = pd.read_csv('paper_values.csv')

# Combine the datasets
combined_df = pd.concat([df1, df2], ignore_index=True, sort=False)

# Replace NaN values with empty strings if any columns didn't match
combined_df = combined_df.fillna('')

# Save the combined DataFrame back to a new CSV file
combined_df.to_csv('paper_values.csv', index=False)
