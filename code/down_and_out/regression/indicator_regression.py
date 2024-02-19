import os
import sys

# Adjust the system path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from generate_data.find_strictly_increase import find_strictly_increase
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

csvFilePath = 'acc_data.csv'
#csvFilePath = 'data.csv'

# Define your model function here; this is an example of a quadratic function.
def model_func(x, a, b, c):
    return a * x**2 + b * x + c

### Function for finding strictly increase but with the H_log version instead
def is_decreasing(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def find_strictly_decrease(data, T_val, sigma_val):
    filtered_data = data[(data['sigma'] == sigma_val) & (data['T'] == T_val)]
    sorted_data = filtered_data.sort_values(by='H_log', ascending=False)
    
    if filtered_data.empty:
        return 0
    
    for i in range(len(sorted_data)):
        split_array = sorted_data.iloc[i:]
        mapped_numbers = split_array['error_percent'].astype(float).tolist()
        if is_decreasing(mapped_numbers):
            return sorted_data.iloc[i]['H_log']
            
    return 0

def test():
    filtered_data = pd.read_csv(csvFilePath)
    #filtered_data = data[(data['K'] == 300)]
    T_range = np.arange(0.2, 5.1, 0.05)
    sigma_range = np.arange(0.2, 0.6, 0.05)
    
    h_values = []  # Store H values
    products = []  # Store corresponding product values
    count = 0
    
    for T in T_range:
        T = round(T, 1)
        for sigma in sigma_range:
            sigma = round(sigma, 2)
            H_value = find_strictly_decrease(filtered_data, T, sigma)
            count += 1 
            
            if float((H_value)) > 0:
                h_values.append(H_value)
                products.append(abs(sigma * np.sqrt(T)))

    print(f"Current count is {count}, length of H_array is {len(h_values)}")

    # Convert lists to numpy arrays for curve fitting
    h_values = np.array(h_values)
    products = np.array(products)
    
    # Fit the model to the data
    popt, pcov = curve_fit(model_func, products, h_values)
    
    # Extract the optimized parameters
    a, b, c = popt

    # Print the full equation with parameters
    print(f"Equation of the fitted model: y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}")
    
    # Calculate the R-squared and MSE for the fitted model
    predicted_values = model_func(products, *popt)
    r_squared = r2_score(h_values, predicted_values)
    mse = mean_squared_error(h_values, predicted_values)
    rmse = np.sqrt(mse)

    # Print out the metrics
    print(f'R-squared: {round(r_squared,2)}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

    # Plot the original data
    plt.figure(figsize=(10, 6))
    plt.scatter(products, h_values, color='blue', label='Original Data')

    # Plot the fitted curve
    x_fit = np.linspace(min(products), max(products), 100)
    y_fit = model_func(x_fit, *popt)
    plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')

    # Add titles and labels
    plt.title('H_log as a Function of Product (Sigma*sqrt(T)) with Fitted Curve')
    plt.xlabel('Product (Sigma*sqrt(T))')
    plt.ylabel('H Log')
    plt.legend()
    plt.grid(True)
    plt.show()

test()
