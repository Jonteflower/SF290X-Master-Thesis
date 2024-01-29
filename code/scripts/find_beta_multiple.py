import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from joblib import Parallel, delayed
from generate_data.find_beta import find_optimal_beta
from generate_data.base_data import get_base_variables

def main():
    # Get the base variables
    m, r, _, _, S0, K, _, beta, _, q = get_base_variables()

    # Read the data
    df = pd.read_csv('data.csv')

    # Generate all unique combinations of H, Sigma, and T
    combinations = [(H, sigma, T) for H in df['H'].unique() 
                                    for sigma in df['sigma'].unique() 
                                    for T in df['T'].unique()]

    # Function to process each combination
    def process_combination(H, sigma, T):
        df_subset = df[(df['H'] == H) & (df['sigma'] == sigma) & (df['T'] == T)]
        if not df_subset.empty:
            return find_optimal_beta(df_subset, S0, K, r, q, sigma, m, H)
        return None

    # Parallel computation for each combination
    results = Parallel(n_jobs=-1)(delayed(process_combination)(H, sigma, T) for H, sigma, T in combinations)

    # Filter out None results and convert to a DataFrame
    results = [result for result in results if result is not None]
    results_df = pd.DataFrame(results, columns=['H', 'Beta', 'Error'])

    # Save to CSV
    results_df.to_csv('best_beta.csv', index=False)

    print("Optimal betas calculation completed. Results saved to 'best_beta.csv'.")

if __name__ == "__main__":
    main()
