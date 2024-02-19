import sys
import os

# Add the 'code' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Now you should be able to import from 'equations' and 'generate_data' without issue
from equations.down_and_out_call_exact import down_and_call_book
from equations.adjusted_barrier import adjusted_barrier, adjusted_barrier_custom
from generate_data.base_data import get_base_variables
import seaborn as sns

# Initialize a list to store error calculations
errors = []

# Read file
df = pd.read_csv('data.csv')

for index, row in df.iterrows():
    q = 0
    r= 0.1 
    beta = 0.5826
    m = 50
    
    # Extract necessary values from the row
    T, H, S0, K, sigma, H_percent = row['T'], row['H'], row['S0'], row['K'], row['sigma'], row['H_log']
    
    # Calculate for custom and regular barriers
    H_down, H_up = adjusted_barrier_custom(T, H, S0, K, sigma, m, beta, df)
    
    price_original = row['price_adj']
    price_custom = down_and_call_book(S0, K, T, r, q, sigma, H, H_down, H_up)
    
    price_mc = row['price_iter']
    error_original = round(abs(((price_mc - price_original) / price_mc) * 100), 4)
    error_custom = round(abs(((price_mc - price_custom) / price_mc) * 100), 4)
        
    errors.append({'H_log': H_percent, 'Error': error_original, 'Type': 'Original'})
    errors.append({'H_log': H_percent, 'Error': error_custom, 'Type': 'Custom'})
    
    #print("Price ",  price_custom, "Price_iter ", price_mc,   "H ", H, "Sigma ", sigma,  "T ", T  )
    
# Convert errors list to DataFrame for plotting
df_errors = pd.DataFrame(errors)

# Aggregate errors by H_percent and Type
df_agg = df_errors.groupby(['H_log', 'Type'], as_index=False)['Error'].mean()

# Find the maximum error values for each method
max_error_original = df_agg[df_agg['Type'] == 'Original']['Error'].max()
max_error_custom = df_agg[df_agg['Type'] == 'Custom']['Error'].max()

print(f"Maximum Error - Original Method: {max_error_original:.4f}%")
print(f"Maximum Error - Custom Method: {max_error_custom:.4f}%")

# Step 3: Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_agg, x='H_log', y='Error', hue='Type', marker='o')
plt.title('Aggregate Error vs. H_log')
plt.xlabel('H_log')
plt.ylabel('Error (%)')
plt.legend(title='Method')
plt.gca().invert_xaxis()  # Invert the x-axis to have the largest H_percent start on the left
plt.show()
