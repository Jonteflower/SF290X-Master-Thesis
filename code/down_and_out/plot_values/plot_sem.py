import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("acc_data_3.csv")

# Extracting accuracies and row numbers for the first 200 rows
df_subset = df.head(200)
accuracies = df_subset['accuracy']
row_numbers = range(1, len(df_subset) + 1)

# Plotting
plt.bar(row_numbers, accuracies, color='blue')
plt.xlabel('Row Number')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot (First 200 rows)')
plt.show()