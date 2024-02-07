import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equations.down_and_out_call_MC import down_and_out_call_MC
from generate_data.base_data import get_base_variables, get_exact_values
import time

# Get base variables
m, r, T, sigma, S0, K, trading_days, beta, H_init, q = get_base_variables()
correct_values = get_exact_values()

def main():
    h_values = range(85, 86)
    n = 8*(10**6)
    start = time.time()

    print("Number of steps ", m*n/10**9,  "Billion")
    for H in h_values:
        price = down_and_out_call_MC(m, r, T, sigma, S0, K, H, n)
        price = round(price, 4)
        print("Barrier H:",H, " ", "Price:", price, " ", "Difference:", round(price-correct_values[H],5))
    
    end = time.time()
    print("Time taken per iteration", (round((end-start)/len(h_values))), "seconds")
    
main()
    
    