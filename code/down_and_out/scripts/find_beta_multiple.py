import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from joblib import Parallel, delayed
from generate_data.find_beta import find_optimal_beta
from generate_data.base_data import get_base_variables

def main():
    # Get the base variables
    m, r, _, _, S0, K, _, _, _, q = get_base_variables()

    # Read the data
    df = pd.read_csv('data.csv')

    # Function to process each row of the dataframe
    def process_row(row):
        # Find the best beta for the given row's parameters and exact price
        best_beta, min_error = find_optimal_beta(S0, K, r, q, row['sigma'], m, row['H'], row['T'], row['price_mc'])
        return {'T': row['T'], 'sigma': row['sigma'], 'H': row['H'], 'Best Beta': best_beta, 'Average Error': min_error}

    # Parallel computation for each row in the dataframe
    results = Parallel(n_jobs=-1)(delayed(process_row)(row) for _, row in df.iterrows())

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV in the specified format
    results_df.to_csv('best_beta.csv', index=False)

    print("Optimal betas calculation completed. Results saved to 'best_beta.csv'.")

if __name__ == "__main__":
    main()
