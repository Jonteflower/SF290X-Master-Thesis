#### Import all the needed packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting tool

############  Exponential Regresion
# Define the exponential model for curve fitting
def exponential_model(x, a, b1, b2):
    return a * np.exp(b1 * x[0] + b2 * x[1])

# Assuming df is your DataFrame and it is already loaded with 'H_log', 'Product', and 'best_beta' columns
df = pd.read_csv('acc_data.csv')

# Prepare the data for fitting
X = np.vstack((df['H_log'], df['Product']))  # Stack 'H_log' and 'Product' for curve fitting
y = df['best_beta'].values

# Fit the exponential model
params, _ = curve_fit(exponential_model, X, y)

# Extract the parameters
a_beta, b1_beta, b2_beta = params

# Predict values using the fitted model parameters for the initial data
y_pred_initial = exponential_model(X, a_beta, b1_beta, b2_beta)

# Linear model for Cook's distance - using OLS for simplicity
X_ols = df[['H_log', 'Product']]
X_ols = sm.add_constant(X_ols)  # Add a constant to the predictors
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

# Refit the exponential model with the filtered data
params_filtered, _ = curve_fit(exponential_model, X_filtered, y_filtered)

# Predict values using the new model parameters for the filtered data
y_pred_filtered = exponential_model(X_filtered, *params_filtered)

# Calculate R-squared value for the filtered model
r_squared_filtered = r2_score(y_filtered, y_pred_filtered)

# Create a figure for 3D plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the original data points
ax.scatter(df['H_log'], df['Product'], df['best_beta'], color='blue', label='Original Data')

# Create a grid to evaluate the model and plot the surface
H_log_range = np.linspace(df['H_log'].min(), df['H_log'].max(), 20)
Product_range = np.linspace(df['Product'].min(), df['Product'].max(), 20)
H_log_grid, Product_grid = np.meshgrid(H_log_range, Product_range)
X_grid = np.vstack((H_log_grid.ravel(), Product_grid.ravel()))

# Predict values using the fitted model parameters for the grid
y_pred_grid = exponential_model(X_grid, *params).reshape(H_log_grid.shape)

# Plot the surface
ax.plot_surface(H_log_grid, Product_grid, y_pred_grid, color='orange', alpha=0.7, label='Fitted Exponential Surface')

# Add titles and labels
ax.set_title('3D Exponential Regression')
ax.set_xlabel('H_log')
ax.set_ylabel('Product (Sigma*sqrt(T))')
ax.set_zlabel('Best Beta')

# Show the plot
plt.show()

# Print the equation of the fitted exponential model
print(f"Fitted exponential equation: best_beta = {params[0]:.4f} * exp({params[1]:.4f} * H_log + {params[2]:.4f} * Product)")

# Calculate R-squared value for the model
print( f"Model R² value: {round(r_squared_filtered, 2)}")