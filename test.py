import numpy as np



def regression_beta_engineer(T, sigma, H, S0):
    beta_start = 0.5826
    beta_end = 0.72
    H_end = S0-1  # This is where beta should reach beta_end
    H_log = abs(np.exp(H/S0))
    
    # Determine the start point of the increase using a logistic function of Sigma_sqrt_T
    Sigma_sqrt_T = sigma * np.sqrt(T)
    H_log_start = 1.5609e-01 / (1 + np.exp(-4.5799e+00 * (Sigma_sqrt_T - 4.4876e-01)))
    H_start = 0  # Convert H_log_start to H_start
    
    # Calculate the growth rate 'b' based on actual H values
    b = (np.log(beta_end) - np.log(beta_start)) / (H_end )
    
    # Correcting the calculation of 'a' to ensure it aligns with the boundary condition at H_start
    a = beta_start / np.exp(b * H_start)
    print("a ", a)
    # Calculate the exponential increase
    beta = beta_start  # Default value if outside the modifying range
    
    # Apply the exponential growth based on actual H values
    if H_log > H_log_start:
        beta = a * np.exp(b * H)

    
    print(beta)
    return beta


T = 0.5
sigma = 0.3
H = 94
S0 = 100

regression_beta_engineer(T, sigma, H, 100)
