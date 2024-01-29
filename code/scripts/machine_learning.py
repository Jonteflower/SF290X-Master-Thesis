import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your data
df = pd.read_csv('Beta_values.csv')

# Assuming your dependent variable is 'beta' and independent variables are 'H', 'T', 'sigma'
X = df[['H', 'T', 'sigma']]
y = df['Best Beta']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
