import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Assuming df is already read from your CSV
df = pd.read_csv('acc_data.csv')
###### Indicator regression

# Quadratic model function
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# Logistic model function
def logistic_model(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Logarithmic model function
def logarithmic_model(x, a, b):
    return a + b * np.log(x)

### Function for finding strictly increase but with the H_log version instead
def is_decreasing(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def find_strictly_decrease(data, T_val, sigma_val):
    df = data[(data['sigma'] == sigma_val) & (data['T'] == T_val)]
    sorted_data = df.sort_values(by='H_log', ascending=False)
    
    if df.empty:
        return 0
    
    for i in range(len(sorted_data)):
        split_array = sorted_data.iloc[i:]
        mapped_numbers = split_array['error_percent'].astype(float).tolist()
        if is_decreasing(mapped_numbers):
            return sorted_data.iloc[i]['H_log']
            
    return 0

# Extract unique values of sigma and T from the DataFrame
unique_sigmas = df['sigma'].unique()
unique_Ts = df['T'].unique()
#unique_ms = df['m'].unique()

# Sort the arrays in case order matters for your calculations
unique_sigmas.sort()
unique_Ts.sort()

h_values = []  # Store H values
products = []  # Store corresponding product values
count = 0

# Iterate over all combinations of unique sigma and T values
for T in unique_Ts:
    for sigma in unique_sigmas:
        H_value = find_strictly_decrease(df, T, sigma)
        count += 1

        # Assuming find_strictly_decrease returns a value where positive indicates a valid result
        if float(H_value) > 0:
            h_values.append(H_value)
            products.append(abs(sigma * np.sqrt(T/50)))

print(f"Processed {count} combinations, obtained {len(h_values)} valid H values.")

# Convert lists to numpy arrays for curve fitting
h_values = np.array(h_values)
products = np.array(products)

# Fit the quadratic model to the data
popt_quad, _ = curve_fit(quadratic_model, products, h_values)
predicted_values_quad = quadratic_model(products, *popt_quad)
r_squared_quad = r2_score(h_values, predicted_values_quad)

# Fit the logistic model to the data
popt_logistic, _ = curve_fit(logistic_model, products, h_values, p0=[max(h_values), 1, np.median(products)])
predicted_values_logistic = logistic_model(products, *popt_logistic)
r_squared_logistic = r2_score(h_values, predicted_values_logistic)

# Fit the logarithmic model to the data, excluding non-positive values
positive_products = products[products > 0]
positive_h_values = h_values[products > 0]
popt_log, _ = curve_fit(logarithmic_model, positive_products, positive_h_values)
predicted_values_log = logarithmic_model(positive_products, *popt_log)
r_squared_log = r2_score(positive_h_values, predicted_values_log)

# Values to pass to the indicator function
L_fitted, k_fitted, x0_fitted = popt_logistic

# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(products, h_values, color='blue', label='Original Data')

# Generate x values for prediction
x_fit = np.linspace(min(products), max(products), 100)

# Plot the quadratic fitted curve
y_fit_quad = quadratic_model(x_fit, *popt_quad)
plt.plot(x_fit, y_fit_quad, color='red', label=f'Quadratic Curve (R² = {r_squared_quad:.4f})')

# Plot the logistic fitted curve
y_fit_logistic = logistic_model(x_fit, *popt_logistic)
plt.plot(x_fit, y_fit_logistic, color='green', label=f'Logistic Curve (R² = {r_squared_logistic:.4f})')

# Plot the logarithmic fitted curve (ensure x_fit is positive)
positive_x_fit = x_fit[x_fit > 0]
y_fit_log = logarithmic_model(positive_x_fit, *popt_log)
plt.plot(positive_x_fit, y_fit_log, color='purple', label=f'Logarithmic Curve (R² = {r_squared_log:.4f})')

# Fit the quadratic model to the data
popt_quad, _ = curve_fit(quadratic_model, products, h_values)
predicted_values_quad = quadratic_model(products, *popt_quad)
r_squared_quad = r2_score(h_values, predicted_values_quad)
print(f"Quadratic model: y = {popt_quad[0]:.4e}x^2 + {popt_quad[1]:.4e}x + {popt_quad[2]:.4e} (R² = {r_squared_quad:.4f})")

# Fit the logistic model to the data
popt_logistic, _ = curve_fit(logistic_model, products, h_values, p0=[max(h_values), 1, np.median(products)])
predicted_values_logistic = logistic_model(products, *popt_logistic)
r_squared_logistic = r2_score(h_values, predicted_values_logistic)
print(f"Logistic model: y = {popt_logistic[0]:.4e} / (1 + exp(-{popt_logistic[1]:.4e}(x - {popt_logistic[2]:.4e}))) (R² = {r_squared_logistic:.4f})")

# Fit the logarithmic model to the data
positive_products = products[products > 0]
positive_h_values = h_values[products > 0]
popt_log, _ = curve_fit(logarithmic_model, positive_products, positive_h_values)
predicted_values_log = logarithmic_model(positive_products, *popt_log)
r_squared_log = r2_score(positive_h_values, predicted_values_log)

# Add titles and labels
plt.title('H_log as a Function of Product (Sigma*sqrt(T)) with Fitted Curves')
plt.xlabel('Product (Sigma*sqrt(T))')
plt.ylabel('H_log')
plt.legend()
plt.grid(True)
plt.show()

# Quadratic model: y = -7.3896e-02x^2 + 2.2475e-01x + -5.4974e-03 (R² = 0.9559)
# Logistic model: y = 1.5609e-01 / (1 + exp(-4.5799e+00(x - 4.4876e-01))) (R² = 0.9551)