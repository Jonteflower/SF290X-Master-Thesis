import numpy as np
import pandas as pd
import statsmodels.api as sm

def filter_by_cooks_distance(x, y ):
    threshold=4/len(y)
    
    # Ensure x is a 2D array for the model
    if x.ndim == 1:
        x = sm.add_constant(x.reshape(-1, 1))
    else:
        x = sm.add_constant(x)

    # Fit the OLS model
    model = sm.OLS(y, x).fit()

    # Compute Cook's distance
    infl = model.get_influence()
    cooks_d = infl.cooks_distance[0]

    # Filter based on the threshold
    mask = cooks_d < threshold
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Return filtered data, excluding the constant column added for the regression
    return x_filtered[:, 1:], y_filtered