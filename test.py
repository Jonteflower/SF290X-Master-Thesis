import numpy as np


t_vals = np.arange(1, 5.1, 1)
m = 100
m2 = 50
sigma = 0.3

for t in t_vals:
    val1 = np.sqrt(t/m)
    print(round(val1,3))
