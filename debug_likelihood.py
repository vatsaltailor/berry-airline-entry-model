"""Debug the likelihood calculation"""

import numpy as np
import sys
from berry_model_v2 import load_and_preprocess_data, calculate_profits, calculate_entry_probabilities

# Redirect output to file
sys.stdout = open('debug_output.txt', 'w')
sys.stderr = sys.stdout

# Load data
X, airline_sizes, entry, outcome_0, outcome_1, outcome_2plus = load_and_preprocess_data()

print("Data loaded successfully")
print(f"X shape: {X.shape}")
print(f"Airline sizes: {airline_sizes}")
print(f"Outcomes: N=0: {outcome_0.sum()}, N=1: {outcome_1.sum()}, N>=2: {outcome_2plus.sum()}")

# Test with simple parameters
beta = np.zeros(5)
delta = 0.135

print(f"\nTest parameters:")
print(f"beta: {beta}")
print(f"delta: {delta}")

# Calculate profits
V = calculate_profits(X, airline_sizes, beta)
print(f"\nProfit matrix V:")
print(f"  Shape: {V.shape}")
print(f"  Range: [{V.min():.4f}, {V.max():.4f}]")
print(f"  Mean: {V.mean():.4f}")
print(f"  First market V: {V[0, :]}")

# Calculate probabilities
p0, p1, p2plus = calculate_entry_probabilities(V, delta)
print(f"\nProbabilities:")
print(f"  p0 range: [{p0.min():.6f}, {p0.max():.6f}]")
print(f"  p1 range: [{p1.min():.6f}, {p1.max():.6f}]")
print(f"  p2plus range: [{p2plus.min():.6f}, {p2plus.max():.6f}]")
print(f"  First market: p0={p0[0]:.6f}, p1={p1[0]:.6f}, p2+={p2plus[0]:.6f}, sum={p0[0]+p1[0]+p2plus[0]:.6f}")

# Calculate log-likelihood
eps = 1e-10
p0_clip = np.clip(p0, eps, 1-eps)
p1_clip = np.clip(p1, eps, 1-eps)
p2plus_clip = np.clip(p2plus, eps, 1-eps)

ll = np.sum(outcome_0 * np.log(p0_clip) + outcome_1 * np.log(p1_clip) + outcome_2plus * np.log(p2plus_clip))
print(f"\nLog-likelihood: {ll:.4f}")

# Check for issues
print(f"\nDiagnostics:")
print(f"  Any NaN in p0: {np.any(np.isnan(p0))}")
print(f"  Any NaN in p1: {np.any(np.isnan(p1))}")
print(f"  Any NaN in p2plus: {np.any(np.isnan(p2plus))}")
print(f"  Any zero in p0: {np.any(p0 == 0)}")
print(f"  Any zero in p1: {np.any(p1 == 0)}")
print(f"  Any zero in p2plus: {np.any(p2plus == 0)}")

# Check a few markets
print(f"\nSample markets:")
for i in [0, 100, 500, 1000]:
    actual = 0 if outcome_0[i] else (1 if outcome_1[i] else 2)
    print(f"  Market {i}: actual N={actual}, p0={p0[i]:.4f}, p1={p1[i]:.4f}, p2+={p2plus[i]:.4f}")

sys.stdout.close()
