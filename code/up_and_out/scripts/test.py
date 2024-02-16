
import sys
import os
# Adjust the system path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from generate_data.find_beta import find_optimal_beta

# Assuming find_optimal_beta function signature looks something like this:
# find_optimal_beta(S0, K, r, q, sigma, m, H, T, price)

# Load the data
df = pd.read_csv('data_up_out_beta.csv')
newdf = df.loc[(df['H'] == 150) & (df['T'] == 0.2) & (df['sigma'] == 0.3) ]
print(newdf)