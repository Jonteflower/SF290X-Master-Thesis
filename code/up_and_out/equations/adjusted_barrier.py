import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data.find_strictly_increase import find_strictly_increase

# Fitted beta
def fitted_beta(H):
    return 0.0008218 * H**2 - 0.1434 * H + 6.833

def regression_beta(T, sigma, H, S0, data):
    H_percent = ((S0-H)/S0)*100
    prod = sigma*np.sqrt(T)
    
    ### Todo find a generalised function for this 
    h_increase = find_strictly_increase(data, T, sigma)
    
    beta = 0.5826
    
    #print("H",H, "S0 ", S0, "h_increase ", h_increase)

    # From regresion_exp function
    if  H_percent >= h_increase:
        ### For data.csv
        beta = 0.6467*np.exp(+0.0171*H_percent-0.0950*prod)
        
        #if H_percent < 2:
            #beta = 0.6467*np.exp(-0.0171*H_percent+0.15*prod)
 
        #beta = 0.5826
        
        #print("Chaning to new beta",beta, "For T ", T, "For Sigma ", sigma, "H_percent ", H_percent)

    return beta

# Function to adjust the barrier for discrete monitoring
def adjusted_barrier(T, H, sigma,m, beta):
    # dT should be here, it "is the time between monitoring instants", p.325, also stated in book from michael at p.628
    delta_T = T / m

    # H_adj = H * np.exp( -1* beta * sigma * np.sqrt(T))
    H_adj_down = H * np.exp( -1* beta * sigma * np.sqrt(delta_T))
    H_adj_up = H * np.exp( beta * sigma * np.sqrt(delta_T))
    
    return H_adj_down, H_adj_up

# Function to adjust the barrier for discrete monitoring
def adjusted_barrier_custom(T, H, S0,K, sigma, m, beta, data):
    
    # dT should be here, it "is the time between monitoring instants", p.325, also stated in book from michael at p.628
    delta_T = T / m
    beta = regression_beta(T, sigma, H, S0, data)

    ### adjust the beta value
    H_adj_down = H * np.exp(-1 * beta * sigma * np.sqrt(delta_T))
    H_adj_up = H * np.exp(beta * sigma * np.sqrt(delta_T))

    return H_adj_down, H_adj_up
