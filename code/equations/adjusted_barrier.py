import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fitted beta
def fitted_beta(H):
    return 0.0008218 * H**2 - 0.1434 * H + 6.833

def regression_beta(T, sigma, H):
    #y = 0.12098770567977638 * np.exp(0.014797502249431218 * T + 0.1570070902550931 * sigma + 0.016314411566790647 * H)
    y = 0.22766957440094568 * np.exp(0.016991538832389925 * T + 0.13137363979574074 * sigma + 0.0108938822558202 * H)
    return y


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
        #beta = beta_values[H]
        #beta = fitted_beta(H)
        beta = regression_beta(T, sigma, H)

    ### adjust the beta value
    H_adj_down = H * np.exp(-1 * beta * sigma * np.sqrt(delta_T))
    H_adj_up = H * np.exp(beta * sigma * np.sqrt(delta_T))

    return H_adj_down, H_adj_up
