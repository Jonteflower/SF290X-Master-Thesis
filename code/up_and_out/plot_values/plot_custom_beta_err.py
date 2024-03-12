import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equations.up_and_out_call import up_and_out_call
from equations.adjusted_barrier import adjusted_barrier, adjusted_barrier_custom

def plot_aggregated_errors(csv_file_path):
    # Step 1: Load the data
    df = pd.read_csv(csv_file_path)
    
    # Initialize a list to store error calculations
    errors = []

    for index, row in df.iterrows():
        q = 0
        r= 0.1 
        beta = 0.5826
        m = 50
        
        # Extract necessary values from the row
        T, H, S0, K, sigma, H_percent = row['T'], row['H'], row['S0'], row['K'], row['sigma'], row['H_percent']
        
        # Calculate for custom and regular barriers
        H_down, H_up = adjusted_barrier_custom(T, H, S0, K, sigma, m, beta, df)
        
        price_original = row['price_adj']
        price_custom = up_and_out_call(S0, K, T, r, q, sigma, H_up)
        
        price_mc = row['price_iter']
        error_original = round(abs(((price_mc - price_original) / price_mc) * 100), 2)
        error_custom = round(abs(((price_mc - price_custom) / price_mc) * 100), 2)
            
        errors.append({'H_percent': H_percent, 'Error': error_original, 'Type': 'Original'})
        errors.append({'H_percent': H_percent, 'Error': error_custom, 'Type': 'Custom'})
        
        #print("Price ",  price_custom, "Price_iter ", price_mc,   "H ", H, "Sigma ", sigma,  "T ", T  )
        
    # Convert errors list to DataFrame for plotting
    df_errors = pd.DataFrame(errors)

    # Aggregate errors by H_percent and Type
    df_agg = df_errors.groupby(['H_percent', 'Type'], as_index=False)['Error'].mean()

    # Step 3: Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_agg, x='H_percent', y='Error', hue='Type', marker='o')
    plt.title('Aggregate Error vs. H_percent')
    plt.xlabel('H_percent')
    plt.ylabel('Error (%)')
    plt.legend(title='Method')
    plt.gca().invert_xaxis()  # Invert the x-axis to have the largest H_percent start on the left
    plt.show()
    

# Assuming 'paper_values.csv' is your data file
plot_aggregated_errors('data_up_out_beta.csv')
