def calculate_cost(n_simulations, cost_per_100ms, avg_invocation_time_seconds):
    """
    Calculate the total cost of running n_simulations.

    Parameters:
    n_simulations (int): Number of simulations to run.
    cost_per_100ms (float): Cost for 100 milliseconds in USD.
    avg_invocation_time_seconds (float): Average duration of one function invocation in seconds.

    Returns:
    float: Total cost in USD.
    """
    avg_invocation_time_ms = avg_invocation_time_seconds * 1000  # Convert seconds to milliseconds
    cost_per_ms = cost_per_100ms / 100  # Cost per millisecond

    # Total cost for one invocation
    cost_per_invocation = avg_invocation_time_ms * cost_per_ms

    # Total cost for n_simulations
    total_cost = cost_per_invocation * n_simulations

    return total_cost

# Example usage:
n_simulations = 10000  # Number of simulations
cost_per_100ms = 0.000027200  # Cost for 100 milliseconds in USD
avg_invocation_time_seconds = 70  # Average duration of one function invocation in seconds

total_cost = calculate_cost(n_simulations, cost_per_100ms, avg_invocation_time_seconds)
print(f"Total cost for {n_simulations} simulations: ${total_cost:.6f}")
