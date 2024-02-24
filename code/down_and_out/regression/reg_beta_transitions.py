import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Assuming df is already read from your CSV
df = pd.read_csv('beta_transitions.csv')

# Updated model functions as before
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def logistic_model(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def logarithmic_model(x, a, b):
    return a + b * np.log(x)

# Iterate over transitions and perform regressions
for transition in range(1, 4):  # Only for H_log_1, H_log_2, and H_log_3
    # Prepare your data for the specific transition
    h_values = df[f'H_log_{transition}_end'].dropna().values
    products = (df['sigma'] * np.sqrt(df['T'])).values[:len(h_values)]

    # Fit the models as before
    # Quadratic
    popt_quad, _ = curve_fit(quadratic_model, products, h_values)
    predicted_values_quad = quadratic_model(products, *popt_quad)
    r_squared_quad = r2_score(h_values, predicted_values_quad)

    # Logistic
    popt_logistic, _ = curve_fit(logistic_model, products, h_values, p0=[max(h_values), 1, np.median(products)])
    predicted_values_logistic = logistic_model(products, *popt_logistic)
    r_squared_logistic = r2_score(h_values, predicted_values_logistic)

    # Logarithmic, ensuring positive values for log model fitting
    positive_products = products[products > 0]
    positive_h_values = h_values[products > 0]
    popt_log, _ = curve_fit(logarithmic_model, positive_products, positive_h_values)
    predicted_values_log = logarithmic_model(positive_products, *popt_log)
    r_squared_log = r2_score(positive_h_values, predicted_values_log)

    # Plotting the results for this transition
    plt.figure(figsize=(10, 6))
    plt.scatter(products, h_values, color='blue', label='Original Data')

    x_fit = np.linspace(min(products), max(products), 100)
    y_fit_quad = quadratic_model(x_fit, *popt_quad)
    plt.plot(x_fit, y_fit_quad, color='red', label=f'Quadratic Curve (R² = {r_squared_quad:.4f})')

    y_fit_logistic = logistic_model(x_fit, *popt_logistic)
    plt.plot(x_fit, y_fit_logistic, color='green', label=f'Logistic Curve (R² = {r_squared_logistic:.4f})')

    positive_x_fit = x_fit[x_fit > 0]
    y_fit_log = logarithmic_model(positive_x_fit, *popt_log)
    plt.plot(positive_x_fit, y_fit_log, color='purple', label=f'Logarithmic Curve (R² = {r_squared_log:.4f})')

    plt.title(f'H_log_{transition} as a Function of Product (Sigma*sqrt(T)) with Fitted Curves')
    plt.xlabel('Product (Sigma*sqrt(T))')
    plt.ylabel(f'H_log_{transition}')
    plt.legend()
    plt.grid(True)
    plt.show()