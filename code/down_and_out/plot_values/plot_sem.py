import matplotlib.pyplot as plt
import pandas as pd

### Error is a funcion of the size of H, T, Sigma, 
df = pd.read_csv("acc_data_m_100_50.csv")

# Extracting accuracies and row numbers for the first 200 rows
df_subset = df.head(200)
accuracies = df_subset['accuracy']*100
row_numbers = range(1, len(df_subset) + 1)

# Plotting
plt.bar(row_numbers, accuracies, color='blue')
plt.xlabel('Row Number')
plt.ylabel('Error %')
plt.title('Accuracy Plot (First 200 rows)')
plt.show()