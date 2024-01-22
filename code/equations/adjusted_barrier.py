import numpy as np

# Mapping of H values to their optimal beta values
beta_values = {
    90: 0.5920999999999986,
    91: 0.5968999999999981,
    92: 0.6016999999999976,
    93:0.6079999999999969,
    94: 0.6168999999999959,
    95: 0.628,
    96: 0.642,
    97: 0.658,
    98: 0.676,
    99: 0.701
}

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
