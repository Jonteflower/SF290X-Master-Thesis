import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats


def simulate_path_segment(m, r, T, sigma, S0, K, H, q, segment_n_paths, dt):
    dW = np.sqrt(dt) * np.random.randn(m, segment_n_paths)
    S = np.zeros((m + 1, segment_n_paths))
    S[0, :] = S0
    for i in range(1, m + 1):
        S[i, :] = S[i - 1, :] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * dW[i - 1, :])
        S[i, :] = np.where(S[i, :] < H, 0, S[i, :])  # Apply barrier condition

    payoffs = np.maximum(S[-1, :] - K, 0) * np.exp(-r * T)
    payoffs = np.where(S.any(axis=0), payoffs, 0)
    return payoffs

def price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n, n_jobs=12):
    n_paths = n
    n_steps = m
    dt = T / n_steps

    # Adjust the number of jobs and paths per job
    confidence_level=0.95
    paths_per_job = max(1, round(n_paths // n_jobs))  # Ensure at least one path per job
    
    # Adjust n_jobs if n_paths is smaller
    n_jobs = min(n_jobs, n_paths)

    # Running simulations in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(simulate_path_segment)(
        m, r, T, sigma, S0, K, H, q, paths_per_job, dt) for _ in range(n_jobs))

    # Check if results are valid before concatenation
    if not results or any(len(res) == 0 for res in results):
        print("Error: No valid data returned from parallel tasks.")
        return 0, 0, 0

    # Concatenate results and calculate final values
    all_payoffs = np.concatenate(results)
    option_price = np.mean(all_payoffs)
    std_error = np.std(all_payoffs)
    sem = std_error / np.sqrt(n_paths)

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    confidence_interval = z_score * sem

    return option_price, sem, confidence_interval

# Initialize parameters which we want to simulate
S0 = 200
K = 200 
q = 0
trading_days = 250
h_min = round(0.85 * S0)
h_max = S0
beta = 0.5826
h_values = np.arange(h_min, h_max)
r_values = [0.1]

### M values will increase the total amount of complexity/iterations
m_values = [25, 50, 75, 100, 200]

## With higher T and sigma the error will increase
t_values = np.arange(0.2, 5.1, 0.2)
sigma_values = np.arange(0.2, 0.51, 0.1)

## These are the toal amount of steps 1*10**8 gives a maximum error of 0.004 at worst
## We need a maximum error of 0.001 which is fine if its rounded down from 0.0014 
n = 5*10**7

# Calculate total iterations
total_iterations = len(r_values) * len(h_values) * len(t_values) * len(sigma_values)*len(m_values)
current_iteration = 0

# Name the file for saving data files
filename = 'test_data.csv'

# Initially, set header flag to True
header_flag = True

# Start the timer
start_time = time.time()

# Simulation loop with progress tracking
for m in m_values:
    for r in r_values:
        for H in h_values:
            for T in t_values:
                T_rounded = round(T, 1)
                for sigma in sigma_values:
                    price_info = price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
                    #print("Accuracy ", round(price_info[1],6), " time taken ", time.time()-start_time)
                    current_iteration += 1
                    print(f"Progress: {current_iteration}/{total_iterations} iterations completed.", end='\r', flush=True)

                    # Create a DataFrame for the current iteration
                    df_iteration = pd.DataFrame([{
                        "H": H, 
                        "T": T_rounded, 
                        "S0": S0, 
                        "K": K, 
                        "m": m, 
                        "sigma": sigma, 
                        "price_iter": price_info[0], 
                        "accuracy": round(price_info[1],5)
                    }])

                    # Append the current iteration's DataFrame to the CSV
                    df_iteration.to_csv(filename, mode='a', header=header_flag, index=False)
                    # Set header flag to False after the first iteration
                    header_flag = False

# Print elapsed time
elapsed_time = time.time() - start_time
print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
print(f"Simulation data saved to '{filename}'")


"""
Test code with these parameters and compare to true price from Glasserman report 
m = 50    # number of time steps
r = 0.1   # risk-free rate
T = 0.2      # time to maturity
sigma = 0.3 # variance of the underlying asset
S0 = 100   # initial stock price
K = 100    # strike price
#H = 85     # barrier level
trading_days = 250 # Number of trading days
beta = 0.5826 # Constant value on beta
#h_values = range(85, 100)
q = 0
"""