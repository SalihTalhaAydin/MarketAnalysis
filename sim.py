import random


def simulate_trades(num_trades, start_capital, win_prob, gain, loss):
    """
    Simulates trades based on win probability, gain, and loss amounts.

    Args:
        num_trades (int): The maximum number of trades to simulate.
        start_capital (float): The initial capital amount.
        win_prob (float): The probability of a winning trade (0.0 to 1.0).
        gain (float): The amount gained on a winning trade.
        loss (float): The amount lost on a losing trade.

    Returns:
        float: The final capital after the simulation.
    """
    capital = start_capital
    for i in range(num_trades):
        if random.random() < win_prob:  # win
            capital += gain
            # outcomes.append(gain) # Removed as outcomes are not used
        else:  # loss
            capital -= loss
            # outcomes.append(-loss) # Removed as outcomes are not used
        # Early stop if account is blown up
        if capital <= 0:
            break
    return capital


# --- Simulation Parameters ---
num_trades = 10000      # Number of trades per simulation run
start_capital = 250     # Starting capital for each simulation
simulation_amount = 1000  # Number of simulation runs per strategy

# --- Strategy Definitions ---
# Each strategy defines win probability and corresponding gain/loss amounts.
strategies = [
    {"name": "1:50", "win_prob": 0.98, "gain": 1, "loss": 50},
    {"name": "1:2",  "win_prob": 0.60, "gain": 2, "loss": 1},
    {"name": "1:3",  "win_prob": 0.40, "gain": 3, "loss": 1},
    {"name": "1:1",  "win_prob": 0.50, "gain": 1, "loss": 1},
    {"name": "1:4",  "win_prob": 0.35, "gain": 4, "loss": 1},
    {"name": "1:5",  "win_prob": 0.30, "gain": 5, "loss": 1},
    {"name": "1:10", "win_prob": 0.20, "gain": 10, "loss": 1}
]

# --- Run Simulations ---
# Run multiple simulations for each strategy for statistical relevance.
results = []
for strategy in strategies:
    final_caps = []
    # Run the simulation multiple times for statistical relevance
    for _ in range(simulation_amount):
        final_cap = simulate_trades(
            num_trades,
            start_capital,
            strategy["win_prob"],
            strategy["gain"],
            strategy["loss"]
        )
        final_caps.append(final_cap)

    results.append({"name": strategy["name"], "final_caps": final_caps})

# --- Analyze and Sort Results ---
# Sort strategies based on the maximum capital achieved in any simulation run.
# Note: Sorting by average or median might give different insights.
sorted_results = sorted(
    results,
    key=lambda x: max(x["final_caps"]),
    reverse=True
)

# --- Output Results ---
print(f"--- Simulation Results ({simulation_amount} runs per strategy) ---")
for result in sorted_results:
    print(f"\nStrategy {result['name']}:")
    print(f"  Max Capital Reached: {max(result['final_caps']):.2f}")
    avg_capital = sum(result['final_caps']) / len(result['final_caps'])
    print(f"  Average Final Capital: {avg_capital:.2f}")
