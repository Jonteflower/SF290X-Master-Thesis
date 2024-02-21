import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Define the logistic model function for curve fitting
def logistic_model(X, L, k, x0):
    x1, x2 = X
    return L / (1 + np.exp(-k * (x1 + x2 - x0)))

# Load your data
# Make sure you have the 'H_log', 'Product', and 'best_beta' columns in your DataFrame
df = pd.read_csv('acc_data.csv')  # Uncomment this line if you have the CSV file

# Prepare the data for fitting
X = np.vstack((df['H_log'], df['Product']))  # Stack 'H_log' and 'Product' for curve fitting
y = df['best_beta'].values

# Initial guess for curve fitting
# Let's assume the following initial guesses:
# L: somewhere in the middle of the expected range for beta (you mentioned 0.58 to 0.71)
# k: just a starting guess, 1 is a common choice
# x0: median of the H_log values
initial_guess = [np.mean([0.58, 0.71]), 1, np.median(df['H_log'])]

# Bounds for the parameters:
# L: We'll set a lower bound a little below the minimum expected value and an upper bound a little above the maximum expected value.
# k: We'll allow a wide range of positive values.
# x0: We'll set a wide range around the initial guess for x0.
bounds = ([0.5, 0, X[0].min()], [0.8, 10, X[0].max()])

# Fit the logistic model within the specified bounds
params, cov = curve_fit(logistic_model, X, y, p0=initial_guess, bounds=bounds, maxfev=80000)

# Extract the parameters
L_fitted, k_fitted, x0_fitted = params

# Predict values using the fitted model parameters
y_pred_initial = logistic_model(X, L_fitted, k_fitted, x0_fitted)
# Predict values using the fitted model parameters
y_pred_initial = logistic_model(X, *params)

# Calculate R-squared value for the initial model
r_squared_initial = r2_score(y, y_pred_initial)

# Prepare data for OLS: Adding a constant to the predictors for Cook's distance
X_ols = sm.add_constant(X.T)  # .T to transpose for OLS format
y_ols = df['best_beta']

# Fit OLS model
model_ols = sm.OLS(y_ols, X_ols).fit()

# Calculate Cook's distance
influence = model_ols.get_influence()
cooks_d = influence.cooks_distance[0]

# Identify points with high Cook's distance
high_cooks_d_indices = np.where(cooks_d > 4 / len(X_ols))[0]

# Filter out the influential points identified by high Cook's distance
df_filtered = df.drop(index=df.index[high_cooks_d_indices]).reset_index(drop=True)
X_filtered = np.vstack((df_filtered['H_log'], df_filtered['Product']))
y_filtered = df_filtered['best_beta'].values

# Set the bounds for the parameters; make sure they are reasonable
bounds = ([0, 0, 0], [max(y_filtered), 100, np.max(X_filtered[0])])

# Refit the logistic model with the filtered data and updated bounds
try:
    params_filtered, cov_filtered = curve_fit(logistic_model, X_filtered, y_filtered, p0=initial_guess, maxfev=80000, bounds=bounds)
except RuntimeError as e:
    print(e)

# Predict values using the new model parameters for the filtered data
y_pred_filtered = logistic_model(X_filtered, *params_filtered)

# Calculate R-squared value for the filtered model
r_squared_filtered = r2_score(y_filtered, y_pred_filtered)

# Print the equation of the fitted logistic model and R-squared values
print(f"Fitted logistic equation: best_beta = {params_filtered[0]:.4f} / (1 + exp(-{params_filtered[1]:.4f} * (H_log + Product - {params_filtered[2]:.4f})))")
print(f"Initial Logistic Model R² value: {r_squared_initial:.4f}")
print(f"Filtered Logistic Model R² value: {r_squared_filtered:.4f}")
