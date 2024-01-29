#import functions_framework
import numpy as np
from scipy import stats

#@functions_framework.http
def price_http(request):
    """HTTP Cloud Function to calculate option price.
    Args:
        request (flask.Request): The request object.
    Returns:
        JSON response with calculated option price, standard error, and confidence interval, or zeros in case of error.
    """
    # Reading the request
    request_json = request.get_json(silent=True)

    # Check if request_json is provided and has necessary parameters
    if request_json and all(k in request_json for k in ('m', 'r', 'T', 'sigma', 'S0', 'K', 'H', 'q', 'confidence_level', 'n_paths')):
        try:
            m = int(request_json['m'])
            r = float(request_json['r'])
            T = float(request_json['T'])
            sigma = float(request_json['sigma'])
            S0 = float(request_json['S0'])
            K = float(request_json['K'])
            H = float(request_json['H'])
            q = float(request_json['q'])
            confidence_level = float(request_json['confidence_level'])
            n_paths = int(request_json['n_paths'])
        except (ValueError, TypeError):
            # Return zeros in case of any conversion error
            return {"option_price": 0, "standard_error": 0, "confidence_interval": 0}
    else:
        # Return zeros if any parameter is missing
        return {"option_price": 0, "standard_error": 0, "confidence_interval": 0}

    # Calculation logic
    n_steps = m
    dt = T / n_steps

    dW = np.sqrt(dt) * np.random.randn(n_steps, n_paths)
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    for i in range(1, n_steps + 1):
        S[i, :] = S[i - 1, :] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * dW[i - 1, :])
        S[i, :] = np.where(S[i, :] < H, 0, S[i, :])

    payoffs = np.maximum(S[-1, :] - K, 0) * np.exp(-r * T)
    payoffs = np.where(S.any(axis=0), payoffs, 0)

    option_price = np.mean(payoffs)
    std_error = np.std(payoffs)
    sem = std_error / np.sqrt(n_paths)

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    confidence_interval = z_score * sem

    # Constructing the response
    response = {
        "option_price": option_price,
        "standard_error": sem,
        "confidence_interval": confidence_interval
    }

    return response
