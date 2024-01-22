def get_base_variables():
    # Example usage:
    m = 50    # number of time steps
    r = 0.1   # risk-free rate
    T = 0.2      # time to maturity
    sigma = 0.3 # variance of the underlying asset
    S0 = 100   # initial stock price
    K = 100    # strike price
    #B = 85     # barrier level
    trading_days = 250 # Number of trading days
    beta = 0.5826 # Constant value on beta
    #h_values = range(85, 100)
    H_init = 85
    q = 0
    
    return m, r, T, sigma, S0, K, trading_days, beta, H_init, q



def get_beta_values():
    # Mapping of H values to their optimal beta values
    beta_values = {
        85: 0.5826,
        86: 0.5826,
        87: 0.5826,
        88: 0.5826,
        89: 0.5826,
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
    return beta_values


