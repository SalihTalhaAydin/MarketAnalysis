import random


def simulate_trades(num_trades, start_capital, win_prob, gain, loss):
    capital = start_capital
    outcomes = []
    for i in range(num_trades):
        if random.random() < win_prob:  # win
            capital += gain
            outcomes.append(gain)
        else:
            capital -= loss
            outcomes.append(-loss)
        # Early stop if account is blown up
        if capital <= 0:
            break
    return capital, outcomes


# Simulation parameters
num_trades = 10000  # simulation count
start_capital = 250
simlation_amount = 1000

# Strategy definitions:
strategies = [
    {"name": "1:50", "win_prob": 0.98, "gain": 1, "loss": 50},
    {"name": "1:2",  "win_prob": 0.60, "gain": 2, "loss": 1},
    {"name": "1:3",  "win_prob": 0.40, "gain": 3, "loss": 1},
    {"name": "1:1",  "win_prob": 0.50, "gain": 1, "loss": 1},
    {"name": "1:4",  "win_prob": 0.35, "gain": 4, "loss": 1},
    {"name": "1:5",  "win_prob": 0.30, "gain": 5, "loss": 1},
    {"name": "1:10", "win_prob": 0.20, "gain": 10, "loss": 1}
]

# Running 15 simulations for each strategy
results = []
for strategy in strategies:
    final_caps = []
    for _ in range(simlation_amount):  # run each strategy 15 times
        final_cap, _ = simulate_trades(num_trades,
                                       start_capital,
                                       strategy["win_prob"],
                                       strategy["gain"],
                                       strategy["loss"])
        final_caps.append(final_cap)

    results.append({"name": strategy["name"], "final_caps": final_caps})

# Sorting results by final capital (from best to worst)
sorted_results = sorted(results, key=lambda x: max(
    x["final_caps"]), reverse=True)

# Output the sorted results
for result in sorted_results:
    print(f"Strategy {result['name']}:")
    print(f"Max Capital after {simlation_amount} simulations: "
          f"{max(result['final_caps'])}")
    avg_capital = sum(result['final_caps']) / len(result['final_caps'])
    print(f"Average Capital: {avg_capital}\n")
