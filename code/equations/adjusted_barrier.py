import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data.data import get_beta_values

beta_values = get_beta_values()
# Function to adjust the barrier for discrete monitoring
def adjusted_barrier(T, H, sigma,m, beta):

    # dT should be here, it "is the time between monitoring instants", p.325, also stated in book from michael at p.628
    delta_T = T / m

    # H_adj = H * np.exp( -1* beta * sigma * np.sqrt(T))
    H_adj_down = H * np.exp( -1* beta * sigma * np.sqrt(delta_T))
    H_adj_up = H * np.exp( beta * sigma * np.sqrt(delta_T))
    
    return H_adj_down, H_adj_up

# Function to adjust the barrier for discrete monitoring
def adjusted_barrier_custom(T, H, S0,K, sigma, m, beta):
    
    # dT should be here, it "is the time between monitoring instants", p.325, also stated in book from michael at p.628
    delta_T = T / m

    # Calculate the new Beta for the final 5% of the values
    if abs(((S0-H)/S0)) <= 0.1:
        beta = beta_values[H]
        #print("New beta is ", beta)

    ### adjust the beta value
    H_adj_down = H * np.exp(-1 * beta * sigma * np.sqrt(delta_T))
    H_adj_up = H * np.exp(beta * sigma * np.sqrt(delta_T))

    return H_adj_down, H_adj_up
