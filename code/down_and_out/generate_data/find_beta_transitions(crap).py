import pandas as pd
import numpy as np

def is_increasing(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def find_strictly_increase(data, T_val, sigma_val):
    filtered_data = data[(data['sigma'] == sigma_val) & (data['T'] == T_val)]
    sorted_data = filtered_data.sort_values(by='H', ascending=False)
    
    if filtered_data.empty:
        return (0, 0)  # Return a tuple for consistency
    
    for i in range(len(sorted_data)):
        split_array = sorted_data.iloc[i:]
        mapped_numbers = split_array['best_beta'].astype(float).tolist()
        
        if is_increasing(mapped_numbers):
            return (sorted_data.iloc[i]['H'], 100 * (data['S0'].max() - sorted_data.iloc[i]['H']) / data['S0'].max())
            
    return (0, 0)  # Return a tuple for consistency

def compute_transitions_all(df, T, sigma):
    
    ##
    beta_values = [0.589, 0.611, 0.689, 0.711]
    transition_data = {'T': T, 'sigma': sigma, 'S0': df['S0'].max()}
    
    for idx, beta in enumerate(beta_values, start=1):
        df_filtered = df[(df['best_beta'] >= beta) & (df['T'] == T) & (df['sigma'] == sigma)]
        if df_filtered.empty:
            print(f"Skipping beta={beta} for T={T}, sigma={sigma} due to no matching data.")
            continue  # Skip this iteration if no data matches the filter
        
        H_log_end, H_percent = find_strictly_increase(df_filtered, T, sigma)
        # Ensure H_log_start calculation is only done if df_filtered is not empty
        H_log_start = df_filtered['H'].iloc[0] if not df_filtered.empty else None
        
        transition_data[f'beta_{idx}'] = beta
        transition_data[f'H_log_{idx}_start'] = abs(np.log(H_log_start/df['S0'].max())) if H_log_start else None
        transition_data[f'H_log_{idx}_end'] = abs(np.log(H_log_end/df['S0'].max())) if H_log_end else None
        transition_data[f'H_percent_{idx}'] = H_percent
    
    return transition_data

# Read the data
file = 'acc_data_3.csv'  # Ensure this is the correct file name
df = pd.read_csv(file)
df = df.round(3)

# Get unique T and sigma combinations
unique_combinations = df[['T', 'sigma']].drop_duplicates()

transitions_list = []

# Iterate over each unique combination and compute transitions
for index, row in unique_combinations.iterrows():
    transition_data = compute_transitions_all(df, row['T'], row['sigma'])
    transitions_list.append(transition_data)

transitions_df = pd.DataFrame(transitions_list)

# Save to CSV
filename = 'beta_transitions_1000.csv'
transitions_df.to_csv(filename, index=False)
print(f"Transitions saved to {filename}.")
