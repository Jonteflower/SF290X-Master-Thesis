import sys
import os
import random
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data.base_data import get_base_variables
from equations.up_and_out_call_Brown import price_up_and_out_call_brown
from generate_data.find_beta import find_optimal_beta 

# Initial values
h_values = range(85, 100)
t_values = [i * 0.1 for i in range(2, 51)]
sigma_values = [i * 0.05 for i in range(4, 13)]
m_values = [5, 25, 50]  # Additional layer for m values

def main():
    # Get base variables
    _, r, _, _, S0, K, trading_days, _, _, q = get_base_variables()  # Removed m from here
    n = 10**6

    # Store the samples and results
    samples = []
    total_iterations = len(h_values) * len(m_values) * 30  # 30 for the number of T and sigma combinations
    current_iteration = 0

    for H in h_values:
        for m in m_values:  # Additional loop for m
            sampled_ts = random.sample(t_values,40)
            sampled_sigmas = random.sample(sigma_values, 7)
            for T, sigma in zip(sampled_ts, sampled_sigmas):
                # Calculate exact price
                price, _, _ = price_up_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)

                # Find optimal beta 
                best_beta, _ = find_optimal_beta(S0, K, r, q, sigma, m, H, T, price)

                samples.append({
                    'S0': S0, 'K': K, 'r': r, 'm': m, 'T': T, 'H': H, 'sigma': sigma, 
                    'trading_days': trading_days, 'exact_price': price, 
                    'best_beta': best_beta  # Replace with actual beta
                })

                current_iteration += 1
                print(f"Progress: {current_iteration}/{total_iterations} iterations completed", end='\r')

    # Convert to DataFrame and save
    samples_df = pd.DataFrame(samples).round(4)
    samples_df.to_csv('sample_data.csv', index=False)
    print("\nData saved to 'sample_data.csv'.")

if __name__ == "__main__":
    main()
