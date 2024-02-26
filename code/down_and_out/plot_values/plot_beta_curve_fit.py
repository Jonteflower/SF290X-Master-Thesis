import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def polynomial(x, *coeffs):
    """A general polynomial function for curve fitting."""
    return sum(c * x**i for i, c in enumerate(coeffs))

def fit_and_plot(data_file):
    # Read the data from a CSV file
    df = pd.read_csv(data_file)

    # Prepare the data for curve fitting
    xdata = df['H']
    ydata = df['best_beta']

    # Guess initial polynomial coefficients (e.g., for a quadratic curve)
    initial_guess = [1, 1, 1]  # Adjust this based on the expected polynomial degree

    # Perform the curve fitting
    coeffs, _ = curve_fit(polynomial, xdata, ydata, p0=initial_guess)

    # Print the coefficients
    print("Polynomial coefficients:", coeffs)

    # Generate fitted values
    fitted_ydata = polynomial(xdata, *coeffs)

    # Plot the original data points
    plt.scatter(xdata, ydata, label='Actual Data', color='black')

    # Plot the fitted curve
    plt.plot(xdata, fitted_ydata, label='Fitted Curve', color='red')

    # Add labels and title
    plt.xlabel('H')
    plt.ylabel('Best Beta')
    plt.title('Curve Fitting for Best Beta Values')
    plt.legend()

    # Show the plot
    plt.show()

# Call the function with the path to your CSV file
fit_and_plot('acc_data_3.csv')
