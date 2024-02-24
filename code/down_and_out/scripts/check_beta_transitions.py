import pandas as pd
import numpy as np

def is_increasing(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def find_strictly_increase(data, T_val, sigma_val):
    filtered_data = data[(data['sigma'] == sigma_val) & (data['T'] == T_val)]
    sorted_data = filtered_data.sort_values(by='H', ascending=False)
    
    if filtered_data.empty:
        return 0
    
    for i in range(len(sorted_data)):
        split_array = sorted_data.iloc[i:]
        mapped_numbers = split_array['best_beta'].astype(float).tolist()
        
        if is_increasing(mapped_numbers):
            return round((sorted_data.iloc[i]['H']), 2)
            
    return 0

def print_best_beta(df, T, sigma, H):
    # Filter the DataFrame based on T, sigma, and H_log
    filtered_df = df[(df['T'] == T) & (df['sigma'] == sigma) & (df['H'] == H)]
    
    # Check if any rows match the given criteria
    if not filtered_df.empty:
        # Print the "best_beta" value
        print(f"For T={T}, sigma={sigma}, and H={H}, best_beta is {filtered_df['best_beta'].iloc[0]}")
    else:
        print(f"No data found for T={T}, sigma={sigma}, and H_log={H}")

# Read the data
file = 'acc_data_3.csv'
df = pd.read_csv(file)

# Filter out best_beta < 0.55
#df = df[(df['best_beta'] >= 0.55) & (df['K'] == 100)]
df = df.round(3)
H_lim = df['S0'].max()

# Example usage:
T_value = 3
sigma_value = 0.35
H_values = np.arange(0.85*H_lim,H_lim, 1)

for H_value in H_values:
    print_best_beta(df, T_value, sigma_value, H_value)

# Points to examine 0.589 being constant, then increase happens to 0.611 which is constant again, then increase to 0.689being constant, then incrase to 0.711 being constant again
H_log_first_start = find_strictly_increase(df, T_value, sigma_value)
print("H_log_1_start: ", 0.85*H_lim)
print("H_log_1_end: ", H_log_first_start)

# First find the value of H when it stops being 0.0589
df_2 = df[df['best_beta'] >= 0.611]
H_log_2_start = df_2.iloc[0]['H']
H_log_2_end = find_strictly_increase(df_2, T_value, sigma_value)
print("H_log_2_start: ", H_log_2_start)
print("H_log_2_end: ", H_log_2_end)

# when it starts being 0.689
df_3 = df[df['best_beta'] >= 0.689]
H_log_3_start = df_3.iloc[0]['H']
H_log_3_end = find_strictly_increase(df_3, T_value, sigma_value)
print("H_log_3_start: ", H_log_3_start)
print("H_log_3_end: ", H_log_3_end)

# when it starts being 0.711
df_4 = df[df['best_beta'] >= 0.71]
H_log_4_start = df_4.iloc[0]['H']
H_log_4_end = find_strictly_increase(df_4, T_value, sigma_value)
print("H_log_4_start: ", H_log_4_start)
print("H_log_4_end: ", H_log_4_end)
