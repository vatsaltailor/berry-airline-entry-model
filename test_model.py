"""Test script to debug the Berry model"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Load data
data = pd.read_csv('Berry_data.csv')
airlines = ['airlineaa', 'airlinedl', 'airlineua', 'airlineal', 'airlinelcc', 'airlinewn']

# Extract entry decisions
entry_decisions = data[airlines].values
n_entrants = entry_decisions.sum(axis=1)

print(f"Number of markets: {len(data)}")
print(f"Entry distribution:")
print(f"  N=0: {(n_entrants == 0).sum()}")
print(f"  N=1: {(n_entrants == 1).sum()}")
print(f"  N>=2: {(n_entrants >= 2).sum()}")

# Check data ranges
print(f"\nData ranges:")
print(f"  Population1: {data['population1'].min():.0f} to {data['population1'].max():.0f}")
print(f"  Population2: {data['population2'].min():.0f} to {data['population2'].max():.0f}")
print(f"  Distance: {data['distance'].min():.0f} to {data['distance'].max():.0f}")
print(f"  Passengers: {data['passengers'].min():.0f} to {data['passengers'].max():.0f}")

# Transform variables
data['log_pop1'] = np.log(data['population1'])
data['log_pop2'] = np.log(data['population2'])
data['log_distance'] = np.log(data['distance'])
data['log_passengers'] = np.log(data['passengers'])

print(f"\nLog-transformed ranges:")
print(f"  Log Pop1: {data['log_pop1'].min():.2f} to {data['log_pop1'].max():.2f}")
print(f"  Log Pop2: {data['log_pop2'].min():.2f} to {data['log_pop2'].max():.2f}")
print(f"  Log Distance: {data['log_distance'].min():.2f} to {data['log_distance'].max():.2f}")
print(f"  Log Passengers: {data['log_passengers'].min():.2f} to {data['log_passengers'].max():.2f}")

# Standardize
data['log_pop1_std'] = (data['log_pop1'] - data['log_pop1'].mean()) / data['log_pop1'].std()
data['log_pop2_std'] = (data['log_pop2'] - data['log_pop2'].mean()) / data['log_pop2'].std()
data['log_distance_std'] = (data['log_distance'] - data['log_distance'].mean()) / data['log_distance'].std()
data['log_passengers_std'] = (data['log_passengers'] - data['log_passengers'].mean()) / data['log_passengers'].std()

print(f"\nStandardized ranges:")
print(f"  Log Pop1 std: {data['log_pop1_std'].min():.2f} to {data['log_pop1_std'].max():.2f}")
print(f"  Log Pop2 std: {data['log_pop2_std'].min():.2f} to {data['log_pop2_std'].max():.2f}")
print(f"  Log Distance std: {data['log_distance_std'].min():.2f} to {data['log_distance_std'].max():.2f}")
print(f"  Log Passengers std: {data['log_passengers_std'].min():.2f} to {data['log_passengers_std'].max():.2f}")

# Test profit calculation with simple parameters
X = np.column_stack([
    np.ones(len(data)),
    data['log_pop1_std'],
    data['log_pop2_std'],
    data['tourism1'],
    data['tourism2'],
    data['log_distance_std'],
    data['log_passengers_std']
])

beta_test = np.array([1.0, 0.5, 0.5, 0.1, 0.1, -0.3, 0.2])
base_profit = X @ beta_test

print(f"\nBase profit range: {base_profit.min():.2f} to {base_profit.max():.2f}")

# Add airline sizes
airline_sizes = {'airlineaa': 1000, 'airlinedl': 1500, 'airlineua': 1000, 
                 'airlineal': 1500, 'airlinelcc': 500, 'airlinewn': 500}

V = np.zeros((len(data), 6))
for i, airline in enumerate(airlines):
    V[:, i] = base_profit + airline_sizes[airline] / 1000.0

print(f"Profit matrix V range: {V.min():.2f} to {V.max():.2f}")

# Test probability calculation
delta = 0.5
V_test = V[0, :]
print(f"\nTest market 0:")
print(f"  V values: {V_test}")
print(f"  Actual entry: {entry_decisions[0]}")

# P(N=0)
p0 = np.prod(norm.cdf(-V_test))
print(f"  P(N=0) = {p0:.6f}")

# P(N=1)
max_idx = np.argmax(V_test)
prob_max_enters = norm.cdf(V_test[max_idx])
prob_others_out = np.prod([norm.cdf(-(V_test[j] - delta)) for j in range(6) if j != max_idx])
p1 = prob_max_enters * prob_others_out
print(f"  P(N=1) = {p1:.6f}")

# P(N>=2)
p2 = 1 - p0 - p1
print(f"  P(N>=2) = {p2:.6f}")
print(f"  Sum = {p0 + p1 + p2:.6f}")

