import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('Beta_values.csv')

# Prepare the data
X = data[['T', 'sigma', 'H']]
y = data['Best Beta']

# Since we're going to use a linear model, we need to transform the target variable
# taking the natural log of y
y_transformed = np.log(y)

# Create and fit the linear regression model on the transformed data
model = LinearRegression()
model.fit(X, y_transformed)

# Extract the model parameters
a_log = model.intercept_
b = model.coef_

# Transform back the intercept to get the 'a' parameter
a = np.exp(a_log)

# Print the model equation
print("Multi-variable exponential model equation:")
print(f"y = {a} * exp({b[0]} * T + {b[1]} * sigma + {b[2]} * H)")

# Predict and transform back the predictions
y_pred_transformed = model.predict(X)
y_pred = np.exp(y_pred_transformed)

# Plotting
plt.scatter(y, y_pred)
plt.xlabel('Actual Best Beta')
plt.ylabel('Predicted Best Beta')
plt.title('Actual vs. Predicted Best Beta')
plt.show()
