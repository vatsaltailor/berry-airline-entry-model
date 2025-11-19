"""Final Berry model estimation with complete results"""

import numpy as np
from berry_model_v2 import load_and_preprocess_data, calculate_profits, calculate_entry_probabilities, negative_log_likelihood
from scipy.optimize import minimize

# Load data
print("="*70)
print(" "*20 + "BERRY AIRLINE ENTRY MODEL")
print(" "*15 + "Static Entry Game Estimation - Berry (1992)")
print("="*70)

print("\nLoading data...")
X, airline_sizes, entry, outcome_0, outcome_1, outcome_2plus = load_and_preprocess_data()

n_markets = X.shape[0]
n_params = X.shape[1]

print(f"Markets: {n_markets}")
print(f"Parameters: {n_params}")
print(f"Outcome distribution: N=0: {outcome_0.sum():.0f}, N=1: {outcome_1.sum():.0f}, N>=2: {outcome_2plus.sum():.0f}")

# Starting values
theta0 = np.zeros(n_params + 1)
theta0[0] = 0.0
theta0[-1] = -2.0

print(f"\nStarting values: {theta0}")

# Test starting likelihood
ll0 = -negative_log_likelihood(theta0, X, airline_sizes, outcome_0, outcome_1, outcome_2plus)
print(f"Starting log-likelihood: {ll0:.4f}")

# Optimize
print("\nOptimizing with L-BFGS-B...")
result = minimize(
    negative_log_likelihood,
    theta0,
    args=(X, airline_sizes, outcome_0, outcome_1, outcome_2plus),
    method='L-BFGS-B',
    options={'maxiter': 1000, 'disp': False}
)

print(f"Optimization complete: {result.success}")
print(f"Message: {result.message}")

# Extract results
theta_hat = result.x
beta_hat = theta_hat[:n_params]
delta_hat = np.exp(theta_hat[n_params])
ll_final = -result.fun

# Calculate predictions
V = calculate_profits(X, airline_sizes, beta_hat)
p0, p1, p2plus = calculate_entry_probabilities(V, delta_hat)

predicted = np.argmax(np.column_stack([p0, p1, p2plus]), axis=1)
actual = np.argmax(np.column_stack([outcome_0, outcome_1, outcome_2plus]), axis=1)

overall_accuracy = np.mean(predicted == actual)

# Accuracy by outcome
acc_0 = np.mean(predicted[actual == 0] == 0) if np.sum(actual == 0) > 0 else 0
acc_1 = np.mean(predicted[actual == 1] == 1) if np.sum(actual == 1) > 0 else 0
acc_2 = np.mean(predicted[actual == 2] == 2) if np.sum(actual == 2) > 0 else 0

# Display results
print("\n" + "="*70)
print("ESTIMATION RESULTS")
print("="*70)

param_names = ['Constant (α)', 'Market Size (β₁)', 'Tourism (β₂)', 
               'Distance (β₃)', 'Passengers (β₄)']

print("\nParameter Estimates:")
print("-" * 70)
print(f"{'Parameter':<30} {'Estimate':>15}")
print("-" * 70)
for i, name in enumerate(param_names):
    print(f"{name:<30} {beta_hat[i]:>15.6f}")
print(f"{'Competition Effect (δ)':<30} {delta_hat:>15.6f}")
print("-" * 70)

print(f"\nLog-Likelihood: {ll_final:.4f}")
print(f"Number of Observations: {n_markets}")
print(f"Number of Parameters: {n_params + 1}")

print("\n" + "="*70)
print("KEY PARAMETERS (3 main estimates as requested)")
print("="*70)
print(f"1. Market Size Effect (β₁):      {beta_hat[1]:>12.6f}")
print(f"2. Distance Effect (β₃):          {beta_hat[3]:>12.6f}")
print(f"3. Competition Effect (δ):        {delta_hat:>12.6f}")
print("="*70)

print("\n" + "="*70)
print("PREDICTION ACCURACY")
print("="*70)
print(f"Overall Accuracy: {100*overall_accuracy:.2f}%")
print(f"\nBy Outcome:")
print(f"  N=0 (No Entry):     {100*acc_0:.2f}% ({np.sum(actual==0):.0f} markets)")
print(f"  N=1 (Monopoly):     {100*acc_1:.2f}% ({np.sum(actual==1):.0f} markets)")
print(f"  N>=2 (Competition): {100*acc_2:.2f}% ({np.sum(actual==2):.0f} markets)")
print("="*70)

# Save comprehensive results
with open('FINAL_RESULTS.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write(" "*20 + "BERRY AIRLINE ENTRY MODEL\n")
    f.write(" "*15 + "Static Entry Game Estimation - Berry (1992)\n")
    f.write("="*70 + "\n\n")
    
    f.write("MODEL SPECIFICATION\n")
    f.write("-"*70 + "\n")
    f.write("Profit Function: π_{i,m} = V_{i,m} - δ * N_{-i,m} + ε_{i,m}\n")
    f.write("where V_{i,m} = α + β'X_m + airline_size_i\n")
    f.write("ε_{i,m} ~ N(0,1)\n\n")
    
    f.write("DATA\n")
    f.write("-"*70 + "\n")
    f.write(f"Number of Markets: {n_markets}\n")
    f.write(f"Number of Airlines: 6 (AA, DL, UA, AL, LCC, WN)\n")
    f.write(f"Outcome Distribution:\n")
    f.write(f"  N=0 (No Entry):     {outcome_0.sum():.0f} markets ({100*outcome_0.sum()/n_markets:.1f}%)\n")
    f.write(f"  N=1 (Monopoly):     {outcome_1.sum():.0f} markets ({100*outcome_1.sum()/n_markets:.1f}%)\n")
    f.write(f"  N>=2 (Competition): {outcome_2plus.sum():.0f} markets ({100*outcome_2plus.sum()/n_markets:.1f}%)\n\n")
    
    f.write("PARAMETER ESTIMATES\n")
    f.write("-"*70 + "\n")
    for i, name in enumerate(param_names):
        f.write(f"{name:<30} {beta_hat[i]:>15.6f}\n")
    f.write(f"{'Competition Effect (δ)':<30} {delta_hat:>15.6f}\n")
    f.write("-"*70 + "\n\n")
    
    f.write("MODEL FIT\n")
    f.write("-"*70 + "\n")
    f.write(f"Log-Likelihood: {ll_final:.4f}\n")
    f.write(f"Overall Prediction Accuracy: {100*overall_accuracy:.2f}%\n")
    f.write(f"  N=0 Accuracy: {100*acc_0:.2f}%\n")
    f.write(f"  N=1 Accuracy: {100*acc_1:.2f}%\n")
    f.write(f"  N>=2 Accuracy: {100*acc_2:.2f}%\n")
    f.write("-"*70 + "\n\n")
    
    f.write("INTERPRETATION\n")
    f.write("-"*70 + "\n")
    f.write(f"1. Market Size (β₁ = {beta_hat[1]:.4f}):\n")
    f.write(f"   Negative coefficient suggests larger markets may have other\n")
    f.write(f"   characteristics that reduce profitability (after controlling for passengers).\n\n")
    f.write(f"2. Distance (β₃ = {beta_hat[3]:.4f}):\n")
    f.write(f"   Positive coefficient indicates longer routes are more profitable.\n\n")
    f.write(f"3. Passengers (β₄ = {beta_hat[4]:.4f}):\n")
    f.write(f"   Strong positive effect - more passengers increase profitability.\n\n")
    f.write(f"4. Competition Effect (δ = {delta_hat:.4f}):\n")
    f.write(f"   Each additional competitor reduces profit by {delta_hat:.4f} units.\n")
    f.write("="*70 + "\n")

print("\nResults saved to FINAL_RESULTS.txt")
print("\n" + "="*70)
print("ESTIMATION COMPLETE!")
print("="*70)

