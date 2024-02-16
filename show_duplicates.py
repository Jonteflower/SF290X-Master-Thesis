import pandas as pd

# Step 1: Load your dataset
df = pd.read_csv('paper_values.csv')

# Step 2: Filter for K=300
df_filtered = df[df['K'] == 300]

# Step 3: Select only the 'T' and 'sigma' columns
df_unique_combinations = df_filtered[['T', 'sigma']]

# Step 4: Drop duplicate rows to get unique combinations
df_unique_combinations = df_unique_combinations.drop_duplicates()

# Step 5: Print the unique combinations
print(df_unique_combinations)



##### Filter duplicates 
def drop_duplicate_rows(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Drop duplicate rows based on columns 'H', 'S0', 'T', and 'sigma'
    df = df.drop_duplicates(subset=['H', 'S0', 'T', 'sigma'], keep='first')

    # Reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)

    return df

drop_duplicate_rows("paper_values.csv")