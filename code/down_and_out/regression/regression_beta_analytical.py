import numpy as np

def regression_beta_analytical(T, sigma, H, S0, m):
    H_log = abs(np.log(H/S0))
    prod = sigma * np.sqrt(T)
    beta_values = [0.5826, 0.6174, 0.6826, 0.7174]

    # Initialize beta to the lowest value by default
    beta = beta_values[0]
    
    # Define regression equations based on the product of sigma and sqrt(T)
    eq_2 = 0.1371 * prod   # End of Beta 0.5826
    eq_3 = 0.0897 * prod   # Start of Beta 0.6174
    eq_4 = 0.0496 * prod   # End of Beta 0.6174
    eq_5 = 0.0467 * prod   # Start of Beta 0.6826
    eq_6 = 0.0207 * prod   # End of Beta 0.6826
    eq_7 = 0.0058 * prod   # Start of Beta 0.7174

    # Linear interpolation between beta values based on conditions involving H_log and prod
    if eq_3 < H_log <= eq_2:
        beta_range = beta_values[1] - beta_values[0]
        beta = ((H_log - eq_3) / (eq_2 - eq_3)) * beta_range + beta_values[0]
    elif eq_4 < H_log <= eq_3:
        beta = beta_values[1]
    elif eq_6 < H_log <= eq_5:
        beta = beta_values[2]
    elif eq_7 < H_log <= eq_6:
        beta_range = beta_values[3] - beta_values[2]
        beta = ((H_log - eq_7) / (eq_6 - eq_7)) * beta_range + beta_values[2]
    elif H_log <= eq_7 and prod >= 0.2:
        beta = beta_values[3]

    return beta

# Function to adjust the barrier for discrete monitoring
def adjusted_barrier_analytical(T, H, S0,K, sigma, m):
    
    # dT should be here, it "is the time between monitoring instants", p.325, also stated in book from michael at p.628
    delta_T = T / m
    beta = regression_beta_analytical(T, sigma, H, S0, m)

    ### adjust the beta value
    H_adj_down = H * np.exp(-1 * beta * sigma * np.sqrt(delta_T))
    H_adj_up = H * np.exp(beta * sigma * np.sqrt(delta_T))

    return H_adj_down, H_adj_up