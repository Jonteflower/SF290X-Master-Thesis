import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.othermod.betareg import BetaModel

# Load your dataset
df = pd.read_csv('acc_data_3.csv')

# Calculate new variables
df['Sigma_sqrt_T'] = df['sigma'] * np.sqrt(df['T'])
df['H_log'] = np.abs(np.log(df['H'] / df['S0']))

# You might want to try different transformations here, 
# for example, square or exponential transformations
df['H_log_squared'] = df['H_log'] ** 2
df['Sigma_sqrt_T_squared'] = df['Sigma_sqrt_T'] ** 2

# Filter the DataFrame based on 'best_beta'
df_filtered = df[df['best_beta'] > 0.58]

# Ensure the dependent variable is within (0, 1)
df_filtered['best_beta'] = df_filtered['best_beta'].clip(lower=0.001, upper=0.999)

# Independent variables with non-linear terms
X = df_filtered[['H_log', 'Sigma_sqrt_T', 'H_log_squared', 'Sigma_sqrt_T_squared']]
X = add_constant(X)  # Adds a constant term to the predictor

# Dependent variable
y = df_filtered['best_beta']

# Fit the beta regression model with non-linear terms
model = BetaModel(y, X)
beta_fit = model.fit()

# Print out the summary of the regression
print(beta_fit.summary())

# Generate predictions using the fitted model
y_pred = beta_fit.predict(X)

# Plot the predictions vs actual values
import matplotlib.pyplot as plt

plt.scatter(df_filtered['Sigma_sqrt_T'], y, label='Actual')
plt.scatter(df_filtered['Sigma_sqrt_T'], y_pred, color='r', label='Predicted')
plt.xlabel('Sigma_sqrt_T')
plt.ylabel('Predicted Beta')
plt.title('Actual vs Predicted Beta Values with Non-linear Model')
plt.legend()
plt.show()
