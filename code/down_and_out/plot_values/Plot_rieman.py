import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta

# Generate a range of values from 0 to 0.8 (excluding 0 because zeta is not defined at 0)
x_values = np.linspace(0.01, 0.8, 400)  # Start from a small value close to 0 to avoid division by zero
y_values = [abs(zeta(x).real) for x in x_values]  # Calculate the real part of the Zeta function

# Calculate the transformed Zeta function values
beta_values = [abs(zeta(x).real / np.sqrt(2 * np.pi)) for x in x_values]

# Beta constant
beta_constant = 0.5826

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x_values, y_values, label="Re(Zeta(x))")
plt.plot(x_values, beta_values, label="Re(Zeta(x)) / sqrt(2π)")
plt.axhline(y=beta_constant, color='gray', linestyle='--', label='Beta = 0.5826')

plt.xlabel('s')
plt.ylabel('Value')
plt.title('Real Part of Riemann Zeta Function and Transformations for 0 < x ≤ 0.8')
plt.legend()
plt.grid(True)
plt.show()
