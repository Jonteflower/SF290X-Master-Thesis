import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the CSV file
df_combined = pd.read_csv('acc_data_3.csv')

# Assuming necessary columns exist
df_combined['Sigma_sqrt_T'] = df_combined['sigma'] * np.sqrt(df_combined['T'])
df_combined = df_combined[df_combined['best_beta'] > 0.68]

# Reset the index to ensure consistency
df_combined.reset_index(drop=True, inplace=True)

# Prepare data for OLS multiple regression: add a constant to the predictors
X = df_combined[['H_log', 'Sigma_sqrt_T']]
X = sm.add_constant(X)
y = df_combined['best_beta']

# Fit OLS multiple regression model
model_ols = sm.OLS(y, X).fit()

# Get summary of the regression
print(model_ols.summary())

# Calculate Cook's distance
influence = model_ols.get_influence()
cooks_d = influence.cooks_distance[0]

# Identify points with high Cook's distance
high_cooks_d_indices = np.where(cooks_d > 4 / len(X))[0]

# Remove these points and refit the model if necessary
df_filtered = df_combined.drop(index=high_cooks_d_indices).reset_index(drop=True)

# Prepare the data for refitting with the filtered dataset
X_filtered = df_filtered[['H_log', 'Sigma_sqrt_T']]
X_filtered = sm.add_constant(X_filtered)
y_filtered = df_filtered['best_beta']

# Refit the multiple regression model with the filtered data
model_ols_filtered = sm.OLS(y_filtered, X_filtered).fit()

# Get summary of the new regression
print(model_ols_filtered.summary())

# Predict values using the new model parameters
y_pred_filtered = model_ols_filtered.predict(X_filtered)

# Plot the results (you can customize this to a 3D plot or other visualization as needed)
plt.scatter(X_filtered['H_log'], y_filtered, label='Data')
plt.scatter(X_filtered['H_log'], y_pred_filtered, label='Fitted', color='red')
plt.xlabel('H_log')
plt.ylabel('best_beta')
plt.legend()
plt.show()

# Print the equation of the fitted model
print(f"Fitted equation: best_beta = {model_ols_filtered.params['const']:.4f} + " \
      f"{model_ols_filtered.params['H_log']:.4f}*H_log + " \
      f"{model_ols_filtered.params['Sigma_sqrt_T']:.4f}*Sigma_sqrt_T")

# Calculate R-squared for the filtered model
print(f"Filtered Model R-squared: {model_ols_filtered.rsquared:.4f}")
