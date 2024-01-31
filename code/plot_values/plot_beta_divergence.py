import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load the dataset
df = pd.read_csv('data_with_beta.csv')

# Define the baseline value and the tolerance for numerical inaccuracies
baseline = 0.5862
tolerance = 0.002  # Adjust as needed based on your data's variability

# Calculate H_percent
df['H_percent'] = ((df['S0'] - df['H']) / df['S0']) * 100

# Function to detect the divergence point from baseline
def find_divergence_point(group):
    # Sort by H_percent just in case
    group.sort_values('H_percent', inplace=True)
    
    # Choose an appropriate window length: smallest odd number less than the size of the group
    window_length = min(51, len(group) - 1)  # make sure window_length is less than the length of the data
    if window_length % 2 == 0:  # if it's even, subtract 1 to make it odd
        window_length -= 1
    if window_length < 3:  # if too small to apply savgol_filter, return NaN
        return np.nan
    
    # Apply a Savitzky-Golay filter to smooth the beta values
    smoothed = savgol_filter(group['best_beta'], window_length=window_length, polyorder=3)
    
    # Calculate the second derivative
    second_derivative = np.gradient(np.gradient(smoothed))
    
    # Find where the second derivative approaches zero after the initial increase
    # We're looking for a stable plateau following an increase
    candidate_points = np.where((np.abs(second_derivative) < tolerance) & (smoothed > baseline))[0]

    # If we found any candidate points, return the first one
    return group.iloc[candidate_points[0]]['H_percent'] if len(candidate_points) > 0 else np.nan

# Collect the divergence points for each T and sigma
divergence_data = []
for (T, sigma), group in df.groupby(['T', 'sigma']):
    divergence_point = find_divergence_point(group)
    if not np.isnan(divergence_point):
        divergence_data.append({
            'Sigma*T': sigma * T,
            'Divergence_H_percent': divergence_point
        })

# Create a DataFrame from the divergence data
divergence_df = pd.DataFrame(divergence_data)

# Plotting the divergence points
plt.figure(figsize=(10, 6))
plt.scatter(divergence_df['Sigma*T'], divergence_df['Divergence_H_percent'], label='Divergence Points')
plt.plot(divergence_df['Sigma*T'], divergence_df['Divergence_H_percent'], label='Trend')

plt.xlabel('Sigma*T')
plt.ylabel('H_percent at Divergence')
plt.title('Divergence from Beta Baseline vs Sigma*T')
plt.legend()
plt.grid(True)
plt.savefig('beta_divergence_plot.png')
plt.show()
