import matplotlib.pyplot as plt
import pandas as pd

### Error is a funcion of the size of H, T, Sigma, 
df = pd.read_csv("acc_data_100.csv")

# Assuming you've already read your DataFrame 'df' from the CSV file
df_subset = df.head(200)

# Ensure 'accuracy' values are converted to float, then multiply by 100
accuracies = df_subset['accuracy'].astype(float) * 100

row_numbers = range(1, len(df_subset) + 1)

# Now that 'accuracies' are floats, the subtraction should work
plt.figure(figsize=(10, 6))  # Optional: Adjusts the figure size
plt.bar(row_numbers, accuracies, color='blue')
plt.xlabel('Row Number')
plt.ylabel('Error %')
plt.title('Accuracy Plot (First 100 rows)')
plt.ylim([min(accuracies)-0.1, max(accuracies)+0.1])  # Adjust Y-axis limits
plt.xticks(rotation=45)  # Optional: Rotates X-axis labels to prevent overlap
plt.tight_layout()  # Adjust layout
plt.show()