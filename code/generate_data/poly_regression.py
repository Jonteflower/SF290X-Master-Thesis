import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('Beta_values.csv')

# Preprocess and prepare the data
X = data[['T', 'sigma', 'H']]
y = data['Best Beta']

# Generating polynomial features
degree = 2  # You can adjust the degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Betas')
plt.ylabel('Predicted Betas')
plt.title('Actual vs Predicted Betas')
#plt.show()

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Fit the model
model.fit(X_train, y_train)

# Coefficients
coefficients = model.coef_
intercept = model.intercept_

# Display the equation
print("Regression equation:")
print(f"y = {intercept}", end=" ")
for i in range(1, len(coefficients)):
    print(f"+ ({coefficients[i]} * x{i})", end=" ")
