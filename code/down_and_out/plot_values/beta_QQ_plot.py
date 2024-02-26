import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('acc_data_3.csv')

# Calculate new variables
df['Sigma_sqrt_T'] = df['sigma'] * np.sqrt(df['T'])
df['H_log'] = np.log(df['H'])

# Filter the DataFrame based on 'best_beta'
df_filtered = df[df['best_beta'] > 0.59]

# Assuming 'best_beta' values are already within the interval [0, 1]
best_beta_values = df_filtered['best_beta']

# Fit the beta and gamma distributions to the 'best_beta' data
a_beta, b_beta, loc_beta, scale_beta = stats.beta.fit(best_beta_values)
a_gamma, loc_gamma, scale_gamma = stats.gamma.fit(best_beta_values)

# Create subplot for the distributions Q-Q plots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjusted for 3 columns and 2 rows

# Q-Q plot for the Beta distribution
stats.probplot(best_beta_values, dist=stats.beta, sparams=(a_beta, b_beta, loc_beta, scale_beta), plot=axs[0, 0])
axs[0, 0].set_title('Q-Q Plot with Beta Distribution')

# Q-Q plot for the Gamma distribution
stats.probplot(best_beta_values, dist=stats.gamma, sparams=(a_gamma, loc_gamma, scale_gamma), plot=axs[0, 1])
axs[0, 1].set_title('Q-Q Plot with Gamma Distribution')

# Q-Q plot for the Normal distribution
stats.probplot(best_beta_values, dist="norm", plot=axs[0, 2])
axs[0, 2].set_title('Q-Q Plot with Normal Distribution')

# Q-Q plot for the Log-Normal distribution
sigma_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(np.log(best_beta_values))
stats.probplot(np.log(best_beta_values), dist=stats.lognorm, sparams=(sigma_lognorm, loc_lognorm, scale_lognorm), plot=axs[1, 0])
axs[1, 0].set_title('Q-Q Plot with Log-Normal Distribution')

# Q-Q plot for the Weibull distribution
c_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(best_beta_values)
stats.probplot(best_beta_values, dist=stats.weibull_min, sparams=(c_weibull, loc_weibull, scale_weibull), plot=axs[1, 1])
axs[1, 1].set_title('Q-Q Plot with Weibull Distribution')

# Q-Q plot for the Poisson distribution
# Poisson is not typically used for continuous data, so this plot is for illustration only
lambda_poisson = np.mean(best_beta_values)
stats.probplot(best_beta_values, dist=stats.poisson, sparams=(lambda_poisson,), plot=axs[1, 2])
axs[1, 2].set_title('Q-Q Plot with Poisson Distribution')

# Display the plots
plt.tight_layout()
plt.show()
