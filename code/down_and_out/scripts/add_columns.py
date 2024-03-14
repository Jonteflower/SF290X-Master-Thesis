import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from equations.adjusted_barrier import (adjusted_barrier)
from equations.down_and_out_call_exact import down_and_call_book
from generate_data.find_beta import find_optimal_beta

def main():
    file_name = 'acc_data_m_75.csv'
    df = pd.read_csv(file_name)

    numeric_columns = ['T', 'm', 'sigma', 'H', 'S0', 'K', 'price_iter']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
            
    def process_row(row):
        best_beta, min_error = find_optimal_beta(row['S0'], row['K'], 0.1, 0, row['sigma'], row['m'], row['H'], row['T'], row['price_iter'])
        H_adj_down, H_adj_up = adjusted_barrier(row['T'], row['H'], row['sigma'], row['m'], 0.5826)
        price = down_and_call_book(row['S0'], row['K'], row['T'], 0.1, 0, row['sigma'], row['H'], row['H'], row['H'])
        price_adj = down_and_call_book(row['S0'], row['K'], row['T'], 0.1, 0, row['sigma'], row['H'], H_adj_down, H_adj_up)
        
        return price, price_adj, best_beta

    # Update the DataFrame directly
    new_data = [process_row(row) for index, row in df.iterrows()]
    
    # Unpack the new data to separate lists
    prices, prices_adj, best_betas = zip(*new_data)

    # Update the existing DataFrame
    df['price'] = prices
    df['price_adj'] = prices_adj
    df['best_beta'] = best_betas

    # Additional calculations if needed
    df['H_log'] = np.abs(np.log(df['H'] / df['S0']))
    df['error_percent'] = np.abs((df['price_iter'] - df['price_adj']) / df['price_iter']) * 100
    df['Product'] = df['sigma'] * np.sqrt(df['T'] / df['m'])
    
    df = df.round(6)

    # Columns to drop
    columns_to_drop = ['price_adj_custom', 'error', 'error_custom']

    # Drop the specified columns
    df_dropped = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Save the updated DataFrame
    df_dropped.to_csv(file_name, index=False)

    print(f"DataFrame updated and saved to {file_name}.")

if __name__ == "__main__":
    main()