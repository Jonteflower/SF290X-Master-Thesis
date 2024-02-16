import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('data.csv')

# Feature creation
data['Sigma_sqrt_T'] = data['sigma'] * np.sqrt(data['T'])
data['log_H_S0'] = np.log(data['H'] / data['S0'])

# Independent variables
X = data[['Sigma_sqrt_T', 'log_H_S0']]
# Dependent variable
y = data['best_beta']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline for standard scaling and ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('ridge_regression', Ridge())
])

# Parameters for GridSearchCV
param_grid = {
    'ridge_regression__alpha': np.logspace(-4, 4, 10)  # Regularization strength
}

# Create and fit the model with GridSearchCV to find the best regularization strength
model = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
model.fit(X_train, y_train)

# Best model
best_model = model.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plotting the results
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual Betas')
plt.ylabel('Predicted Betas')
plt.title('Actual vs Predicted Betas')
plt.show()

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print("Best model parameters:", model.best_params_)

# Note: Displaying the exact regression equation for Ridge regression can be complex due to regularization,
# but you can show the coefficients for insight:
coefficients = best_model.named_steps['ridge_regression'].coef_
intercept = best_model.named_steps['ridge_regression'].intercept_
print("Intercept:", intercept)
print("Coefficients:", coefficients)
