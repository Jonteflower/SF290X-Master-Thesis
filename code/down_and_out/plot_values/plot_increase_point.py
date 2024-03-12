import sys
import os
# Adjust the system path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from generate_data.find_strictly_increase import find_strictly_increase

csvFilePath = 'acc_data_m_100.csv'
#csvFilePath = 'data.csv'

def test():
    data = pd.read_csv(csvFilePath)
    filtered_data = data[(data['K'] == 300)]
    T_range = np.arange(0.2, 5.1, 0.05)
    sigma_range = np.arange(0.2, 0.6, 0.05)
    
    h_values = []  # Store H values
    products = []  # Store corresponding product values
    count = 0
    
    for T in T_range:
        T = round(T, 1)
        for sigma in sigma_range:
            sigma = round(sigma, 2)
            H_value = find_strictly_increase(filtered_data, T, sigma)
            count += 1 
            
            if float((H_value)) > 0:
                print("H_value ", H_value, "T_value ", T, "Sigma_value ", sigma )
                h_values.append(float(abs(np.log(H_value/300))))
                #h_values.append(H_value)
                #products.append(abs(np.log(H_value/300)/(sigma * np.sqrt(T))))
                products.append(abs(sigma * np.sqrt(T)))

    print(f"Current count is {count}, length of H_array is {len(h_values)}",)

    # Fitting a polynomial line (degree 1 for linear fit)
    coeffs = np.polyfit(products, h_values, 4)
    # Creating a polynomial function from the coefficients
    poly = np.poly1d(coeffs)
    print(poly)
    # Generating y-values (fitted values) for plotting
    fitted_values = poly(products)
    
    # Plotting original data
    plt.figure(figsize=(10, 6))
    plt.scatter(products, h_values, color='blue', label='Original Data')
    
    # Plotting fitted line
    plt.plot(products, fitted_values, color='red', label='Fitted Line')
    
    plt.title('H Percent as a Function of Product (Sigma*sqrt(T)) with Fitted Line')
    plt.xlabel('Product (Sigma*sqrt(T))')
    plt.ylabel('H Percent')
    plt.legend()
    plt.grid(True)
    plt.show()

test()