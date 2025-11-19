#!/usr/bin/env python3
"""
Debug version of corrected Berry model
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import traceback

print("="*70)
print("CORRECTED BERRY MODEL - DEBUG VERSION")
print("="*70)

try:
    # Load data
    print("\n1. Loading data...")
    data = pd.read_csv('Berry_data.csv')
    data['passengers'] = data['passengers'].fillna(data['passengers'].median())
    
    airlines = ['airlineaa', 'airlinedl', 'airlineua', 'airlineal', 'airlinelcc', 'airlinewn']
    entry = data[airlines].values
    n_entrants = entry.sum(axis=1)
    
    outcome_0 = (n_entrants == 0).astype(float)
    outcome_1 = (n_entrants == 1).astype(float)
    outcome_2plus = (n_entrants >= 2).astype(float)
    
    print(f"   Markets: {len(data)}")
    print(f"   N=0: {outcome_0.sum():.0f}, N=1: {outcome_1.sum():.0f}, N>=2: {outcome_2plus.sum():.0f}")
    
    # Prepare covariates
    print("\n2. Preparing covariates...")
    population = (data['population1'] + data['population2']) / 2
    log_pop = np.log(population)
    log_dist = np.log(data['distance'])
    log_dist_sq = 2 * log_dist
    
    X = np.column_stack([np.ones(len(data)), log_pop, log_dist, log_dist_sq])
    print(f"   X shape: {X.shape}")
    
    # Define probability calculation
    print("\n3. Defining probability calculations...")
    
    def calc_probs(V, delta):
        n_markets, n_firms = V.shape
        p0 = np.zeros(n_markets)
        p1 = np.zeros(n_markets)

        for m in range(n_markets):
            # P(N=0) - Product of all firms staying out
            p0[m] = np.prod(norm.cdf(-V[m, :]))

            # P(N=1) - CORRECTED FORMULA from R code line 148
            # Sort profits in descending order
            V_sorted = np.sort(V[m, :])[::-1]
            V1 = V_sorted[0]  # Highest profit
            V2 = V_sorted[1]  # Second highest profit

            # Formula: Φ(V1)×Φ(-V1+δ) - [Φ(-V1+δ)-Φ(-V1)]×[Φ(-V2+δ)-Φ(-V2)]×[1-Φ((V2-V1)/2)]
            term1 = norm.cdf(V1) * norm.cdf(-V1 + delta)
            term2 = (norm.cdf(-V1 + delta) - norm.cdf(-V1))
            term3 = (norm.cdf(-V2 + delta) - norm.cdf(-V2))
            term4 = (1 - norm.cdf((V2 - V1) / 2))

            p1[m] = term1 - term2 * term3 * term4

        # P(N>=2)
        p2plus = 1.0 - p0 - p1

        # Clip to ensure valid probabilities
        eps = 1e-10
        p0 = np.clip(p0, eps, 1-eps)
        p1 = np.clip(p1, eps, 1-eps)
        p2plus = np.clip(p2plus, eps, 1-eps)

        # Renormalize
        total = p0 + p1 + p2plus
        return p0/total, p1/total, p2plus/total
    
    # Define likelihood
    def nll(theta, X, y0, y1, y2):
        beta = theta[:X.shape[1]]
        delta = np.exp(theta[X.shape[1]])
        
        V_base = X @ beta
        V = np.tile(V_base[:, np.newaxis], (1, 6))
        
        p0, p1, p2 = calc_probs(V, delta)
        
        eps = 1e-10
        p0 = np.clip(p0, eps, 1-eps)
        p1 = np.clip(p1, eps, 1-eps)
        p2 = np.clip(p2, eps, 1-eps)
        
        ll = np.sum(y0*np.log(p0) + y1*np.log(p1) + y2*np.log(p2))
        return -ll if np.isfinite(ll) else 1e10
    
    # Test initial likelihood
    print("\n4. Testing initial likelihood...")
    theta0 = np.zeros(5)
    theta0[-1] = 0.0  # log(delta) = 0 => delta = 1
    ll0 = nll(theta0, X, outcome_0, outcome_1, outcome_2plus)
    print(f"   Initial NLL: {ll0:.4f}")
    print(f"   Initial delta: {np.exp(theta0[-1]):.4f}")
    
    # Estimate
    print("\n5. Estimating parameters (BFGS)...")
    result = minimize(
        nll, theta0,
        args=(X, outcome_0, outcome_1, outcome_2plus),
        method='BFGS',
        options={'maxiter': 1000, 'disp': True}
    )
    
    print(f"\n6. Optimization complete!")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    print(f"   Iterations: {result.nit}")
    
    # Extract results
    beta = result.x[:4]
    delta = np.exp(result.x[4])
    ll = -result.fun
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nParameter Estimates:")
    print(f"  β₀ (Constant):        {beta[0]:>12.6f}")
    print(f"  β₁ (Log Population):  {beta[1]:>12.6f}")
    print(f"  β₂ (Log Distance):    {beta[2]:>12.6f}")
    print(f"  β₃ (Log Distance²):   {beta[3]:>12.6f}")
    print(f"  δ  (Competition):     {delta:>12.6f}")
    print(f"\nModel Fit:")
    print(f"  Log-Likelihood:       {ll:>12.4f}")
    print("="*70)
    
except Exception as e:
    print(f"\n\nERROR OCCURRED:")
    print(f"{type(e).__name__}: {e}")
    print("\nTraceback:")
    traceback.print_exc()
