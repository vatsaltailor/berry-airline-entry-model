import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# Open file
f = open('quick_test_output.txt', 'w')

f.write("Testing corrected Berry model\n")
f.write("="*70 + "\n")

# Load data
data = pd.read_csv('Berry_data.csv')
data['passengers'] = data['passengers'].fillna(data['passengers'].median())

airlines = ['airlineaa', 'airlinedl', 'airlineua', 'airlineal', 'airlinelcc', 'airlinewn']
entry = data[airlines].values
n_entrants = entry.sum(axis=1)

outcome_0 = (n_entrants == 0).astype(float)
outcome_1 = (n_entrants == 1).astype(float)
outcome_2plus = (n_entrants >= 2).astype(float)

f.write(f"Markets: {len(data)}\n")
f.write(f"N=0: {outcome_0.sum():.0f}, N=1: {outcome_1.sum():.0f}, N>=2: {outcome_2plus.sum():.0f}\n\n")

# Corrected specification
population = (data['population1'] + data['population2']) / 2
log_pop = np.log(population)
log_dist = np.log(data['distance'])
log_dist_sq = 2 * log_dist

X = np.column_stack([np.ones(len(data)), log_pop, log_dist, log_dist_sq])

f.write("Corrected Model Specification:\n")
f.write("  Covariates: intercept, log(pop), log(dist), log(dist²)\n")
f.write(f"  X shape: {X.shape}\n\n")

# Probability calculation
def calc_probs(V, delta):
    n_markets, n_firms = V.shape
    p0 = np.zeros(n_markets)
    p1 = np.zeros(n_markets)
    
    for m in range(n_markets):
        p0[m] = np.prod(norm.cdf(-V[m, :]))
        
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

# Estimate
f.write("Estimating...\n")
theta0 = np.zeros(5)
result = minimize(nll, theta0, args=(X, outcome_0, outcome_1, outcome_2plus),
                  method='BFGS', options={'maxiter': 1000, 'disp': False})

beta = result.x[:4]
delta = np.exp(result.x[4])
ll = -result.fun

f.write("\n" + "="*70 + "\n")
f.write("RESULTS\n")
f.write("="*70 + "\n")
f.write(f"Constant:        {beta[0]:>12.6f}\n")
f.write(f"Log(Population): {beta[1]:>12.6f}\n")
f.write(f"Log(Distance):   {beta[2]:>12.6f}\n")
f.write(f"Log(Distance²):  {beta[3]:>12.6f}\n")
f.write(f"Delta:           {delta:>12.6f}\n")
f.write(f"Log-Likelihood:  {ll:>12.4f}\n")
f.write(f"Converged:       {result.success}\n")
f.write("="*70 + "\n")

f.close()
print("Test complete - see quick_test_output.txt")

