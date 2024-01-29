import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from generate_data.base_data import get_base_variables
from equations.down_and_out_call_Brown import price_down_and_out_call_brown
import time

def main(): 
    # Initial values
    h_values = range(99, 100)
    n = 2 * 10**7
    
    # Get base variables
    m, r, T, sigma, S0, K, trading_days, beta, H, q = get_base_variables()

    # Print the number of steps
    print("Number of steps:", m * n / 10**9, "Billion")
    
    # Start time
    start = time.time()
    
    # Loop all H
    for H in h_values:
        price, sem, conf_interval = price_down_and_out_call_brown(m, r, T, sigma, S0, K, H, q, n)
        lower_bound = price - conf_interval
        upper_bound = price + conf_interval
        print("Barrier H:", H, "Price:", round(price, 3), "SEM:", round(sem, 4), "Confidence Interval:", f"({round(lower_bound, 3)}, {round(upper_bound, 3)})")
    
    # Print results
    time_per_iteration = round((time.time() - start) / len(h_values), 1)
    print("Time/iteration:", time_per_iteration, "s")

main()
    
    