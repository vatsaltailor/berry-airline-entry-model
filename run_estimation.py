"""Run the Berry model estimation and save results"""

import numpy as np
from berry_model_v2 import load_and_preprocess_data, calculate_profits, calculate_entry_probabilities, negative_log_likelihood
from scipy.optimize import minimize

# Load data
print("Loading data...")
X, airline_sizes, entry, outcome_0, outcome_1, outcome_2plus = load_and_preprocess_data()

n_markets = X.shape[0]
n_params = X.shape[1]

print(f"Markets: {n_markets}")
print(f"Parameters: {n_params}")
print(f"Outcomes: N=0: {outcome_0.sum():.0f}, N=1: {outcome_1.sum():.0f}, N>=2: {outcome_2plus.sum():.0f}")

# Starting values
theta0 = np.zeros(n_params + 1)
theta0[0] = 0.0
theta0[-1] = -2.0

print(f"\nStarting values: {theta0}")

# Test starting likelihood
ll0 = -negative_log_likelihood(theta0, X, airline_sizes, outcome_0, outcome_1, outcome_2plus)
print(f"Starting log-likelihood: {ll0:.4f}")

# Optimize
print("\nOptimizing (this may take a few minutes)...")
result = minimize(
    negative_log_likelihood,
    theta0,
    args=(X, airline_sizes, outcome_0, outcome_1, outcome_2plus),
    method='L-BFGS-B',
    options={'maxiter': 1000, 'disp': True}
)

print(f"Optimization complete: {result.success}")
print(f"Message: {result.message}")

# Extract results
theta_hat = result.x
beta_hat = theta_hat[:n_params]
delta_hat = np.exp(theta_hat[n_params])
ll_final = -result.fun

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Constant:        {beta_hat[0]:.6f}")
print(f"Market Size:     {beta_hat[1]:.6f}")
print(f"Tourism:         {beta_hat[2]:.6f}")
print(f"Distance:        {beta_hat[3]:.6f}")
print(f"Passengers:      {beta_hat[4]:.6f}")
print(f"Delta:           {delta_hat:.6f}")
print(f"\nLog-Likelihood:  {ll_final:.4f}")
print("="*60)

# Save to file
with open('estimation_results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("BERRY AIRLINE ENTRY MODEL - ESTIMATION RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Number of Markets: {n_markets}\n")
    f.write(f"Number of Parameters: {n_params + 1}\n\n")
    f.write("Parameter Estimates:\n")
    f.write("-"*60 + "\n")
    f.write(f"Constant (α):              {beta_hat[0]:>12.6f}\n")
    f.write(f"Market Size (β₁):          {beta_hat[1]:>12.6f}\n")
    f.write(f"Tourism (β₂):              {beta_hat[2]:>12.6f}\n")
    f.write(f"Distance (β₃):             {beta_hat[3]:>12.6f}\n")
    f.write(f"Passengers (β₄):           {beta_hat[4]:>12.6f}\n")
    f.write(f"Competition Effect (δ):    {delta_hat:>12.6f}\n")
    f.write("-"*60 + "\n\n")
    f.write(f"Log-Likelihood: {ll_final:.4f}\n")
    f.write(f"Optimization Success: {result.success}\n")
    f.write("="*60 + "\n")

print("\nResults saved to estimation_results.txt")
