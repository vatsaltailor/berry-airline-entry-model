#!/usr/bin/env python3
"""
Berry Airline Entry Model - HYBRID VERSION
Combines:
- Correct P(N=1) formula from R code (theoretical rigor)
- Original covariates including passengers (empirical performance)
- No airline heterogeneity (baseline specification)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

print("="*70)
print("BERRY MODEL - HYBRID VERSION")
print("Correct P(N=1) + Original Covariates")
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
    
    # Prepare covariates - ORIGINAL SPECIFICATION
    print("\n2. Preparing covariates (ORIGINAL SPECIFICATION)...")
    population = (data['population1'] + data['population2']) / 2
    tourism = (data['tourism1'] + data['tourism2']) / 2
    
    # Standardize
    avg_log_pop = np.log(population)
    avg_log_pop = (avg_log_pop - avg_log_pop.mean()) / avg_log_pop.std()
    
    tourism_std = (tourism - tourism.mean()) / tourism.std()
    
    log_dist = np.log(data['distance'])
    log_dist = (log_dist - log_dist.mean()) / log_dist.std()
    
    log_pass = np.log(data['passengers'])
    log_pass = (log_pass - log_pass.mean()) / log_pass.std()
    
    X = np.column_stack([
        np.ones(len(data)),
        avg_log_pop,
        tourism_std,
        log_dist,
        log_pass
    ])
    print(f"   Covariates: intercept, avg_log(pop), tourism, log(dist), log(passengers)")
    print(f"   X shape: {X.shape}")
    
    # Define probability calculation - CORRECTED P(N=1)
    print("\n3. Using CORRECTED P(N=1) formula from R code...")
    
    def calc_probs(V, delta):
        n_markets, n_firms = V.shape
        p0 = np.zeros(n_markets)
        p1 = np.zeros(n_markets)
        
        for m in range(n_markets):
            # P(N=0)
            p0[m] = np.prod(norm.cdf(-V[m, :]))
            
            # P(N=1) - CORRECTED
            V_sorted = np.sort(V[m, :])[::-1]
            V1, V2 = V_sorted[0], V_sorted[1]
            
            term1 = norm.cdf(V1) * norm.cdf(-V1 + delta)
            term2 = (norm.cdf(-V1 + delta) - norm.cdf(-V1))
            term3 = (norm.cdf(-V2 + delta) - norm.cdf(-V2))
            term4 = (1 - norm.cdf((V2 - V1) / 2))
            
            p1[m] = term1 - term2 * term3 * term4
        
        p2plus = 1.0 - p0 - p1
        
        # Clip
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
        V = np.tile(V_base[:, np.newaxis], (1, 6))  # No heterogeneity
        
        p0, p1, p2 = calc_probs(V, delta)
        
        ll = np.sum(y0*np.log(p0) + y1*np.log(p1) + y2*np.log(p2))
        return -ll if np.isfinite(ll) else 1e10
    
    # Test initial likelihood
    print("\n4. Testing initial likelihood...")
    theta0 = np.zeros(6)  # 5 betas + 1 log(delta)
    theta0[-1] = 0.0  # log(delta) = 0 => delta = 1
    ll0 = nll(theta0, X, outcome_0, outcome_1, outcome_2plus)
    print(f"   Initial NLL: {ll0:.4f}")
    
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
    print(f"   Iterations: {result.nit}")
    
    # Extract results
    beta = result.x[:5]
    delta = np.exp(result.x[5])
    ll = -result.fun
    
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
    
    print("\n" + "="*70)
    print("HYBRID MODEL RESULTS")
    print("="*70)
    print(f"\nParameter Estimates:")
    print(f"  Constant (α):         {beta[0]:>12.6f}")
    print(f"  Market Size (β₁):     {beta[1]:>12.6f}")
    print(f"  Tourism (β₂):         {beta[2]:>12.6f}")
    print(f"  Distance (β₃):        {beta[3]:>12.6f}")
    print(f"  Passengers (β₄):      {beta[4]:>12.6f}")
    print(f"  Competition (δ):      {delta:>12.6f}")
    print(f"\nModel Fit:")
    print(f"  Log-Likelihood:       {ll:>12.4f}")
    print(f"  Prediction Accuracy:  {100*accuracy:>11.2f}%")
    print("="*70)
    
except Exception as e:
    print(f"\n\nERROR: {e}")
    import traceback
    traceback.print_exc()

