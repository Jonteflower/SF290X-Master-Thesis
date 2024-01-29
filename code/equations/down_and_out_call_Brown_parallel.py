import numpy as np
import time
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data.data import get_base_variables, get_exact_values
from memory_profiler import profile
from joblib import Parallel, delayed
from memory_profiler import profile

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


def price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n=10**7, confidence_level=0.95, n_jobs=-1):
    n_paths = n
    n_steps = m
    dt = T / n_steps

    # Adjust the number of jobs and paths per job
    n_jobs = 8
    paths_per_job = max(1, n_paths // n_jobs)  # Ensure at least one path per job

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

def main(): 
    # Initial values
    h_values = range(99, 100)
    n = 8 * 10**6
    
    # Get base variables
    m, r, T, sigma, S0, K, trading_days, beta, H, q = get_base_variables()
    correct_values = get_exact_values()

    # Print the number of steps
    print("Number of steps:", m * n / 10**9, "Billion")
    
    # Start time
    start = time.time()
    
    # Loop all H
    for H in h_values:
        price, sem, conf_interval = price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
        lower_bound = price - conf_interval
        upper_bound = price + conf_interval
        print("Barrier H:", H, "Price:", round(price, 3), "SEM:", round(sem, 3), "Confidence Interval:", f"({round(lower_bound, 3)}, {round(upper_bound, 3)})", "Difference:", round(price - correct_values[H], 4))
    
    # Print results
    time_per_iteration = round((time.time() - start) / len(h_values), 1)
    print("Time/iteration:", time_per_iteration, "s")


#main()
    