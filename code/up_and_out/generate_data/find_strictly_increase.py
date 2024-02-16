
def is_increasing(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def find_strictly_increase(data, T_val, sigma_val):
    filtered_data = data[(data['sigma'] == sigma_val) & (data['T'] == T_val)]
    sorted_data = filtered_data.sort_values(by='H_percent', ascending=False)
    
    if filtered_data.empty:
        return 0
    
    for i in range(len(sorted_data)):
        split_array = sorted_data.iloc[i:]
        mapped_numbers = split_array['error_percent'].astype(float).tolist()
        
        if is_increasing(mapped_numbers):
            return round((sorted_data.iloc[i]['H_percent']), 4)
            
    return 0

def is_increasing_log(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def find_strictly_increase_log(data, T_val, sigma_val):
    filtered_data = data[(data['sigma'] == sigma_val) & (data['T'] == T_val)]
    sorted_data = filtered_data.sort_values(by='H_log', ascending=False)
    
    if filtered_data.empty:
        return 0
    
    for i in range(len(sorted_data)):
        split_array = sorted_data.iloc[i:]
        mapped_numbers = split_array['error_percent'].astype(float).tolist()
        
        if is_increasing(mapped_numbers):
            return round((sorted_data.iloc[i]['H_percent']), 4)
            
    return 0