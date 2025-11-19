"""
Compare original and corrected Berry model implementations
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*25 + "BERRY MODEL COMPARISON")
print("="*80)

# Load data
data = pd.read_csv('Berry_data.csv')
data['passengers'] = data['passengers'].fillna(data['passengers'].median())

airlines = ['airlineaa', 'airlinedl', 'airlineua', 'airlineal', 'airlinelcc', 'airlinewn']
entry = data[airlines].values
n_entrants = entry.sum(axis=1)

outcome_0 = (n_entrants == 0).astype(float)
outcome_1 = (n_entrants == 1).astype(float)
outcome_2plus = (n_entrants >= 2).astype(float)

print(f"\nData: {len(data)} markets")
print(f"Outcomes: N=0: {outcome_0.sum():.0f}, N=1: {outcome_1.sum():.0f}, N>=2: {outcome_2plus.sum():.0f}")

# ============================================================================
# MODEL 1: CORRECTED (matching R)
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: CORRECTED SPECIFICATION (Matching R Reference Code)")
print("="*80)

# Covariates: intercept, log(pop), log(dist), log(dist^2)
population = (data['population1'] + data['population2']) / 2
log_pop = np.log(population)
log_dist = np.log(data['distance'])
log_dist_sq = 2 * log_dist

X1 = np.column_stack([np.ones(len(data)), log_pop, log_dist, log_dist_sq])

print(f"Covariates: intercept, log(population), log(distance), log(distance²)")
print(f"X shape: {X1.shape}")

def calc_probs_corrected(V, delta):
    """Corrected probability calculation matching R"""
    n_markets, n_firms = V.shape
    p0 = np.zeros(n_markets)
    p1 = np.zeros(n_markets)
    
    for m in range(n_markets):
        # P(N=0)
        p0[m] = np.prod(norm.cdf(-V[m, :]))
        
        # P(N=1) - R formula
        V_sorted = np.sort(V[m, :])[::-1]
        V1, V2 = V_sorted[0], V_sorted[1]
        
        term1 = norm.cdf(V1) * norm.cdf(-V1 + delta)
        term2 = (norm.cdf(-V1 + delta) - norm.cdf(-V1))
        term3 = (norm.cdf(-V2 + delta) - norm.cdf(-V2))
        term4 = (1 - norm.cdf((V2 - V1) / 2))
        
        p1[m] = term1 - term2 * term3 * term4
    
    p2plus = 1.0 - p0 - p1
    p0 = np.maximum(p0, 0)
    p1 = np.maximum(p1, 0)
    p2plus = np.maximum(p2plus, 0)
    
    total = p0 + p1 + p2plus
    return p0/total, p1/total, p2plus/total

def nll_corrected(theta, X, y0, y1, y2):
    beta = theta[:X.shape[1]]
    delta = np.exp(theta[X.shape[1]])
    
    V_base = X @ beta
    V = np.tile(V_base[:, np.newaxis], (1, 6))
    
    p0, p1, p2 = calc_probs_corrected(V, delta)
    
    eps = 1e-10
    p0 = np.clip(p0, eps, 1-eps)
    p1 = np.clip(p1, eps, 1-eps)
    p2 = np.clip(p2, eps, 1-eps)
    
    ll = np.sum(y0*np.log(p0) + y1*np.log(p1) + y2*np.log(p2))
    return -ll if np.isfinite(ll) else 1e10

# Estimate
theta0_1 = np.zeros(5)  # 4 betas + 1 delta
print("\nEstimating...")
result1 = minimize(nll_corrected, theta0_1, args=(X1, outcome_0, outcome_1, outcome_2plus),
                   method='BFGS', options={'maxiter': 1000, 'disp': False})

beta1 = result1.x[:4]
delta1 = np.exp(result1.x[4])
ll1 = -result1.fun

print(f"\nResults:")
print(f"  Constant:        {beta1[0]:>10.6f}")
print(f"  Log(Population): {beta1[1]:>10.6f}")
print(f"  Log(Distance):   {beta1[2]:>10.6f}")
print(f"  Log(Distance²):  {beta1[3]:>10.6f}")
print(f"  Delta:           {delta1:>10.6f}")
print(f"  Log-Likelihood:  {ll1:>10.4f}")
print(f"  Converged:       {result1.success}")

# ============================================================================
# MODEL 2: ORIGINAL (with issues)
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: ORIGINAL SPECIFICATION (With Issues)")
print("="*80)

# Covariates: intercept, avg log(pop), tourism, log(dist), log(passengers)
pop_avg = (np.log(data['population1']) + np.log(data['population2'])) / 2
pop_avg_std = (pop_avg - pop_avg.mean()) / pop_avg.std()
tourism_total = data['tourism1'] + data['tourism2']
dist_std = (np.log(data['distance']) - np.log(data['distance']).mean()) / np.log(data['distance']).std()
pass_std = (np.log(data['passengers']) - np.log(data['passengers']).mean()) / np.log(data['passengers']).std()

X2 = np.column_stack([np.ones(len(data)), pop_avg_std, tourism_total, dist_std, pass_std])
airline_sizes = np.array([0.1, 0.15, 0.1, 0.15, 0.05, 0.05])

print(f"Covariates: intercept, avg_log(pop), tourism, log(dist), log(passengers)")
print(f"X shape: {X2.shape}")
print(f"Airline sizes: {airline_sizes}")

def calc_probs_original(V, delta):
    """Original (incorrect) probability calculation"""
    n_markets, n_firms = V.shape
    p0 = np.zeros(n_markets)
    p1 = np.zeros(n_markets)
    
    for m in range(n_markets):
        p0[m] = np.prod(norm.cdf(-V[m, :]))
        
        # WRONG: Only considers max firm
        max_idx = np.argmax(V[m, :])
        V_max = V[m, max_idx]
        
        log_p1 = norm.logcdf(V_max)
        for j in range(n_firms):
            if j != max_idx:
                log_p1 += norm.logcdf(-(V[m, j] - delta))
        
        p1[m] = np.exp(log_p1)
    
    p2plus = 1.0 - p0 - p1
    p0 = np.maximum(p0, 0)
    p1 = np.maximum(p1, 0)
    p2plus = np.maximum(p2plus, 0)
    
    total = p0 + p1 + p2plus
    return p0/total, p1/total, p2plus/total

def nll_original(theta, X, airline_sizes, y0, y1, y2):
    beta = theta[:X.shape[1]]
    delta = np.exp(theta[X.shape[1]])
    
    V_base = X @ beta
    V = V_base[:, np.newaxis] + airline_sizes[np.newaxis, :]
    
    p0, p1, p2 = calc_probs_original(V, delta)
    
    eps = 1e-10
    p0 = np.clip(p0, eps, 1-eps)
    p1 = np.clip(p1, eps, 1-eps)
    p2 = np.clip(p2, eps, 1-eps)
    
    ll = np.sum(y0*np.log(p0) + y1*np.log(p1) + y2*np.log(p2))
    return -ll if np.isfinite(ll) else 1e10

# Estimate
theta0_2 = np.zeros(6)  # 5 betas + 1 delta
theta0_2[-1] = -2.0
print("\nEstimating...")
result2 = minimize(nll_original, theta0_2, args=(X2, airline_sizes, outcome_0, outcome_1, outcome_2plus),
                   method='L-BFGS-B', options={'maxiter': 1000, 'disp': False})

beta2 = result2.x[:5]
delta2 = np.exp(result2.x[5])
ll2 = -result2.fun

print(f"\nResults:")
print(f"  Constant:        {beta2[0]:>10.6f}")
print(f"  Market Size:     {beta2[1]:>10.6f}")
print(f"  Tourism:         {beta2[2]:>10.6f}")
print(f"  Distance:        {beta2[3]:>10.6f}")
print(f"  Passengers:      {beta2[4]:>10.6f}")
print(f"  Delta:           {delta2:>10.6f}")
print(f"  Log-Likelihood:  {ll2:>10.4f}")
print(f"  Converged:       {result2.success}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"\nLog-Likelihood Improvement: {ll1 - ll2:+.4f}")
print(f"  Corrected Model: {ll1:.4f}")
print(f"  Original Model:  {ll2:.4f}")
print(f"\nCompetition Effect (δ):")
print(f"  Corrected: {delta1:.6f}")
print(f"  Original:  {delta2:.6f}")
print(f"  Difference: {delta1 - delta2:+.6f}")

print("\n" + "="*80)
print("KEY DIFFERENCES")
print("="*80)
print("1. Probability Calculation:")
print("   - Corrected: Uses R formula accounting for 2nd-highest firm")
print("   - Original:  Only considers highest firm (WRONG)")
print("\n2. Covariates:")
print("   - Corrected: log(pop), log(dist), log(dist²)")
print("   - Original:  avg_log(pop), tourism, log(dist), log(passengers)")
print("\n3. Airline Heterogeneity:")
print("   - Corrected: None (all firms same)")
print("   - Original:  Airline-specific size parameters")
print("="*80)

