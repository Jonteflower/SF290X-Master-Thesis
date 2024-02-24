##################### You need to run the code for adding the beta transitions first from the generate data atb. 
#################### Then move them into the combined beta file ------> TODO Automate this inot one file

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data from CSV file
transitions_df = pd.read_csv('beta_transitions.csv')

# Calculate sigma*sqrt(T) for each row and filter
transitions_df['sigma_sqrt_T'] = transitions_df['sigma'] * np.sqrt(transitions_df['T'])
transitions_df = transitions_df[transitions_df['sigma_sqrt_T'] > 0.21]

# Setup the figure and axes for a 2x4 grid of plots
fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

# Titles for the plots
titles = ['Transition 1', 'Transition 2', 'Transition 3', 'Transition 4']
beta_values = [0.5826, 0.6174, 0.6826, 0.7174]

for i in range(1, 5):  # Loop through all four transitions
    for j, phase in enumerate(['start', 'end']):
        # Filter valid data points for each transition's start and end
        valid = transitions_df.dropna(subset=[f'H_log_{i}_{phase}'])

        # Prepare data for linear regression
        x = valid['sigma_sqrt_T']
        y = valid[f'H_log_{i}_{phase}']
        x_with_constant = sm.add_constant(x)  # Adds a constant term to the predictor

        # Fit the model and calculate Cook's distance
        model = sm.OLS(y, x_with_constant).fit()
        influence = model.get_influence()
        cooks = influence.cooks_distance[0]

        # Identify and remove outliers
        threshold = 4 / len(x)
        non_outliers = cooks < threshold

        # Fit the model without outliers
        model_no_outliers = sm.OLS(y[non_outliers], x_with_constant[non_outliers]).fit()
        print(f"Beta {beta_values[i-1]} {phase.capitalize()} {model_no_outliers.params[1]:.4f}x + {model_no_outliers.params[0]:.4f}, R² = {model_no_outliers.rsquared:.4f}")

        # Plotting
        axs[j, i-1].scatter(x[non_outliers], y[non_outliers], color='blue' if phase == 'start' else 'red', label=f'{phase.capitalize()} (Beta={beta_values[i-1]:.3f}, R²={model_no_outliers.rsquared:.4f})')
        axs[j, i-1].plot(x, x_with_constant.dot(model_no_outliers.params), 'k-', label='Fitted Line')

        # Set legends
        axs[j, i-1].legend()

    # Set titles and labels
    axs[j, i-1].set_title(f'{titles[i-1]}')
    axs[j, i-1].set_xlabel('Sigma * sqrt(T)')
    axs[j, i-1].set_ylabel('H_log')
    axs[j, i-1].grid(True)

# Adjust layout for readability and display the plot
plt.tight_layout()
plt.show()
