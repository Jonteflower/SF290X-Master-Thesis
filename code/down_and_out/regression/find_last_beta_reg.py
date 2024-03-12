import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('acc_data_m_100_50.csv')

# Quadratic model function
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# Logistic model function
def logistic_model(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Logarithmic model function
def logarithmic_model(x, a, b):
    return a + b * np.log(x)

# Extract unique values of 'T' and 'sigma' from the DataFrame
unique_Ts = df['T'].unique()
unique_Sigmas = df['sigma'].unique()
unique_ms = df['m'].unique()

# Sort the arrays in case order matters for your calculations
unique_Ts.sort()
unique_Sigmas.sort()

x_values = []  # Store x values
y_values = []  # Store y values

# Iterate over unique combinations
for m in unique_ms: 
    for T in unique_Ts:
        for sigma in unique_Sigmas:
            # Filter dataframe for the current combination
            subset_df = df[(df['T'] == T) & (df['sigma'] == sigma) & (df['m'] == m)]
            
            # Calculate the product of Sigma and sqrt(T/m)
            x = sigma * np.sqrt(T/m)
            
            # Find the maximum value of 'best_beta' in this subset
            y = subset_df['best_beta'].max()
            
            # Append the values to the lists
            if y > 0.65 and y < 0.8: 
                x_values.append(x)
                y_values.append(y)

print(f"Processed {len(x_values)} combinations.")

# Convert lists to numpy arrays for curve fitting
x_values = np.array(x_values)
y_values = np.array(y_values)

# Fit the quadratic model to the data
popt_quad, _ = curve_fit(quadratic_model, x_values, y_values)
predicted_values_quad = quadratic_model(x_values, *popt_quad)
r_squared_quad = r2_score(y_values, predicted_values_quad)

# Fit the logistic model to the data
popt_logistic, _ = curve_fit(logistic_model, x_values, y_values)
predicted_values_logistic = logistic_model(x_values, *popt_logistic)
r_squared_logistic = r2_score(y_values, predicted_values_logistic)

# Fit the logarithmic model to the data
popt_log, _ = curve_fit(logarithmic_model, x_values, y_values)
predicted_values_log = logarithmic_model(x_values, *popt_log)
r_squared_log = r2_score(y_values, predicted_values_log)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', label='Original Data')

# Generate x values for prediction
x_fit = np.linspace(min(x_values), max(x_values), 100)

# Plot the quadratic fitted curve
y_fit_quad = quadratic_model(x_fit, *popt_quad)
plt.plot(x_fit, y_fit_quad, color='red', label=f'Quadratic Curve (R² = {r_squared_quad:.4f})')

# Plot the logistic fitted curve
y_fit_logistic = logistic_model(x_fit, *popt_logistic)
plt.plot(x_fit, y_fit_logistic, color='green', label=f'Logistic Curve (R² = {r_squared_logistic:.4f})')

# Plot the logarithmic fitted curve
y_fit_log = logarithmic_model(x_fit, *popt_log)
plt.plot(x_fit, y_fit_log, color='purple', label=f'Logarithmic Curve (R² = {r_squared_log:.4f})')


#### Print the model equations
print(f"Logistic model: y = {popt_logistic[0]:.4e} / (1 + exp(-{popt_logistic[1]:.4e}(x - {popt_logistic[2]:.4e}))) (R² = {r_squared_logistic:.4f})")
print(f"Quadratic model: y = {popt_quad[0]:.4e}x^2 + {popt_quad[1]:.4e}x + {popt_quad[2]:.4e} (R² = {r_squared_quad:.4f})")

# Add titles and labels
plt.title('Maximum best_beta vs Sigma * sqrt(T) with Fitted Curves')
plt.xlabel('Sigma * sqrt(T)')
plt.ylabel('Maximum best_beta')
plt.legend()
plt.grid(True)
plt.show()


##  y = 7.0648e-01 / (1 + exp(-3.9661e+00(x - -4.9009e-01))) (R² = 0.9408)
