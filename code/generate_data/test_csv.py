import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier
from equations.down_and_out_call_MC import down_and_out_call_MC
from data import get_base_variables
from equations.down_and_out_call_Brown import price_down_and_out_call_brown

# Get base variables
m, r, T, sigma, S0, K, trading_days, beta, H_init, q = get_base_variables() # Modify as needed
H = 85

def compute_specific_prices():
    # Compute prices
    H_adj_down, H_adj_up = adjusted_barrier(T, H, sigma, m, beta)
    price_mc = round(price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q), 3)
    price = round(down_and_call_book(S0, K, T, r, q, sigma, H, H, H), 3)
    price_adj = round(down_and_call_book(S0, K, T, r, q, sigma, H, H_adj_down, H_adj_up), 3)

    return price_mc, price, price_adj

def read_csv_and_compare(csv_file='simulation_results.csv'):
    # Load DataFrame from CSV
    df = pd.read_csv(csv_file)

    # Filter the DataFrame for the specific values
    filtered_df = df[(df['H'] == H) & (df['sigma'] == sigma) & (df['r'] == r) & (df['T'] == T)]

    # Print the values from the CSV if the row exists
    if not filtered_df.empty:
        print("Values from CSV:")
        print(filtered_df.iloc[0])
    else:
        print("No matching data found in CSV for the provided parameters.")

# Example usage
if __name__ == "__main__":

    # Compute and print prices
    price_mc, price, price_adj = compute_specific_prices()
    print(f"Monte Carlo Price: {price_mc}")
    print(f"Price: {price}")
    print(f"Adjusted Price: {price_adj}")

    # Read from CSV and compare
    read_csv_and_compare()
