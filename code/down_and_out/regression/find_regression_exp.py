import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

# Load the CSV file
df_combined = pd.read_csv('updated_accurate_values.csv')

# For simplicity, using the whole dataset as is, assuming necessary columns exist
df_combined['Sigma_sqrt_T'] = df_combined['sigma'] * np.sqrt(df_combined['T'])

# Define the exponential model for curve fitting
def exponential_model(X, a, b1, b2):
    H_log, sigma_sqrt_t = X
    return a * np.exp(b1 * H_log + b2 * sigma_sqrt_t)

# Prepare the data for fitting
X = (df_combined['H_log'].values, df_combined['Sigma_sqrt_T'].values)
y = df_combined['best_beta'].values

# Fit the exponential model
params, covariance = curve_fit(exponential_model, X, y)

# Extract the parameters
a, b1, b2 = params

# Evaluation and plotting code is omitted for brevity; see below for the Cook's distance part

# Linear model for Cook's distance - using OLS for simplicity
# Prepare data for OLS: here, we choose 'H_log' and 'Sigma_sqrt_T' as predictors
X_ols = df_combined[['H_log', 'Sigma_sqrt_T']]
y_ols = df_combined['best_beta']

# Add a constant to the predictors
X_ols = sm.add_constant(X_ols)

# Fit OLS model
model_ols = sm.OLS(y_ols, X_ols).fit()

# Calculate Cook's distance
influence = model_ols.get_influence()
cooks_d = influence.cooks_distance[0]

# Plot Cook's distance
plt.figure(figsize=(10, 6))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", use_line_collection=True)
plt.title("Cook's Distance")
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.show()

# Identify points with high Cook's distance
high_cooks_d_indices = np.where(cooks_d > 4 / len(X_ols))[0]
print("Indices with high Cook's distance:", high_cooks_d_indices)

# Optional: Remove these points and refit the model if necessary
# Step 1: Remove influential points
df_filtered = df_combined.drop(high_cooks_d_indices).reset_index(drop=True)

# Prepare the data for fitting again, this time with the filtered dataset
X_filtered = (df_filtered['H_log'].values, df_filtered['Sigma_sqrt_T'].values)
y_filtered = df_filtered['best_beta'].values

# Step 2: Refit the exponential model with the filtered data
params_filtered, covariance_filtered = curve_fit(exponential_model, X_filtered, y_filtered)

# Extract the parameters for the new model
a_filtered, b1_filtered, b2_filtered = params_filtered

# Print the equation of the fitted exponential model
print(f"Fitted exponential equation: Beta = {a:.4f} * exp({b1:.4f} * H_log + {b2:.4f} * sigma_sqrt_T)")

# Predict values using the new model parameters
y_pred_filtered = exponential_model(X_filtered, a_filtered, b1_filtered, b2_filtered)

# Step 3: Evaluate the new model
# Since R² is not directly provided by curve_fit, we calculate it manually
ss_res_filtered = np.sum((y_filtered - y_pred_filtered) ** 2)
ss_tot_filtered = np.sum((y_filtered - np.mean(y_filtered)) ** 2)
r_squared_filtered = 1 - (ss_res_filtered / ss_tot_filtered)

print(f"Filtered Model R² value: {r_squared_filtered:.4f}")
