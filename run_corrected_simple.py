#!/usr/bin/env python3
"""
Simple script to run corrected Berry model and save results
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# Open output file
outfile = open('corrected_simple_output.txt', 'w', buffering=1)

def log_print(msg):
    """Print to both file and ensure it's written"""
    outfile.write(msg + '\n')
    outfile.flush()

log_print("="*70)
log_print("CORRECTED BERRY MODEL ESTIMATION")
log_print("="*70)

# Load data
log_print("\n1. Loading data...")
data = pd.read_csv('Berry_data.csv')
data['passengers'] = data['passengers'].fillna(data['passengers'].median())

airlines = ['airlineaa', 'airlinedl', 'airlineua', 'airlineal', 'airlinelcc', 'airlinewn']
entry = data[airlines].values
n_entrants = entry.sum(axis=1)

outcome_0 = (n_entrants == 0).astype(float)
outcome_1 = (n_entrants == 1).astype(float)
outcome_2plus = (n_entrants >= 2).astype(float)

log_print(f"   Markets: {len(data)}")
log_print(f"   N=0: {outcome_0.sum():.0f}, N=1: {outcome_1.sum():.0f}, N>=2: {outcome_2plus.sum():.0f}")

# Prepare covariates (CORRECTED)
log_print("\n2. Preparing covariates (CORRECTED SPECIFICATION)...")
population = (data['population1'] + data['population2']) / 2
log_pop = np.log(population)
log_dist = np.log(data['distance'])
log_dist_sq = 2 * log_dist  # log(dist^2) = 2*log(dist)

X = np.column_stack([np.ones(len(data)), log_pop, log_dist, log_dist_sq])
log_print(f"   Covariates: intercept, log(pop), log(dist), log(dist²)")
log_print(f"   X shape: {X.shape}")

# Define probability calculation (CORRECTED)
log_print("\n3. Defining probability calculations (CORRECTED P(N=1))...")

def calc_probs(V, delta):
    n_markets, n_firms = V.shape
    p0 = np.zeros(n_markets)
    p1 = np.zeros(n_markets)
    
    for m in range(n_markets):
        # P(N=0)
        p0[m] = np.prod(norm.cdf(-V[m, :]))
        
        # P(N=1) - CORRECTED FORMULA
        V_sorted = np.sort(V[m, :])[::-1]
        V1, V2 = V_sorted[0], V_sorted[1]
        
        term1 = norm.cdf(V1) * norm.cdf(-V1 + delta)
        term2 = (norm.cdf(-V1 + delta) - norm.cdf(-V1))
        term3 = (norm.cdf(-V2 + delta) - norm.cdf(-V2))
        term4 = (1 - norm.cdf((V2 - V1) / 2))
        
        p1[m] = term1 - term2 * term3 * term4
    
    p2plus = 1.0 - p0 - p1
    
    # Ensure probabilities are valid
    p0 = np.maximum(p0, 0)
    p1 = np.maximum(p1, 0)
    p2plus = np.maximum(p2plus, 0)
    
    total = p0 + p1 + p2plus
    return p0/total, p1/total, p2plus/total

# Define likelihood
def nll(theta, X, y0, y1, y2):
    beta = theta[:X.shape[1]]
    delta = np.exp(theta[X.shape[1]])
    
    V_base = X @ beta
    V = np.tile(V_base[:, np.newaxis], (1, 6))  # All firms same (NO HETEROGENEITY)
    
    p0, p1, p2 = calc_probs(V, delta)
    
    eps = 1e-10
    p0 = np.clip(p0, eps, 1-eps)
    p1 = np.clip(p1, eps, 1-eps)
    p2 = np.clip(p2, eps, 1-eps)
    
    ll = np.sum(y0*np.log(p0) + y1*np.log(p1) + y2*np.log(p2))
    return -ll if np.isfinite(ll) else 1e10

# Estimate
log_print("\n4. Estimating parameters (BFGS optimizer)...")
theta0 = np.zeros(5)  # 4 betas + 1 log(delta)

result = minimize(
    nll, theta0,
    args=(X, outcome_0, outcome_1, outcome_2plus),
    method='BFGS',
    options={'maxiter': 1000, 'disp': False}
)

# Extract results
beta = result.x[:4]
delta = np.exp(result.x[4])
ll = -result.fun

log_print("\n" + "="*70)
log_print("RESULTS")
log_print("="*70)
log_print(f"\nParameter Estimates:")
log_print(f"  β₀ (Constant):        {beta[0]:>12.6f}")
log_print(f"  β₁ (Log Population):  {beta[1]:>12.6f}")
log_print(f"  β₂ (Log Distance):    {beta[2]:>12.6f}")
log_print(f"  β₃ (Log Distance²):   {beta[3]:>12.6f}")
log_print(f"  δ  (Competition):     {delta:>12.6f}")
log_print(f"\nModel Fit:")
log_print(f"  Log-Likelihood:       {ll:>12.4f}")
log_print(f"  Converged:            {result.success}")
log_print(f"  Iterations:           {result.nit}")

# Calculate predictions
V_base = X @ beta
V = np.tile(V_base[:, np.newaxis], (1, 6))
p0, p1, p2 = calc_probs(V, delta)

pred = np.zeros(len(data))
pred[p0 > np.maximum(p1, p2)] = 0
pred[p1 > np.maximum(p0, p2)] = 1
pred[p2 > np.maximum(p0, p1)] = 2

actual = np.zeros(len(data))
actual[outcome_0 == 1] = 0
actual[outcome_1 == 1] = 1
actual[outcome_2plus == 1] = 2

accuracy = np.mean(pred == actual)
log_print(f"  Prediction Accuracy:  {100*accuracy:>11.2f}%")

log_print("\n" + "="*70)
log_print("VALIDATION COMPLETE")
log_print("="*70)

outfile.flush()
outfile.close()
print("Done! Results in corrected_simple_output.txt")
