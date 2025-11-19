"""
Berry Airline Entry Model - CORRECTED Implementation
Following the R reference code exactly

This corrects the probability calculations to match the R implementation
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_path='Berry_data.csv'):
    """
    Load and preprocess data to match R specification
    """
    data = pd.read_csv(data_path)
    
    # Handle missing values
    data['passengers'] = data['passengers'].fillna(data['passengers'].median())
    
    # Airline names (order: AA, DL, UA, AL, LCC, WN)
    airlines = ['airlineaa', 'airlinedl', 'airlineua', 'airlineal', 'airlinelcc', 'airlinewn']
    
    # Entry decisions
    entry = data[airlines].values  # 2742 x 6
    n_entrants = entry.sum(axis=1)
    
    # Outcome indicators
    outcome_0 = (n_entrants == 0).astype(float)
    outcome_1 = (n_entrants == 1).astype(float)
    outcome_2plus = (n_entrants >= 2).astype(float)
    
    # Market fundamentals - MATCH R SPECIFICATION
    # R uses: intercept, log(population), log(distance), log(distance^2)
    
    # Average population (matching R's single population variable)
    population = (data['population1'] + data['population2']) / 2
    log_pop = np.log(population)
    
    # Distance terms
    log_dist = np.log(data['distance'])
    log_dist_sq = np.log(data['distance'] ** 2)  # This is 2*log(distance)
    
    # Construct X matrix to match R: intercept, log(pop), log(dist), log(dist^2)
    X = np.column_stack([
        np.ones(len(data)),
        log_pop,
        log_dist,
        log_dist_sq
    ])
    
    return X, entry, outcome_0, outcome_1, outcome_2plus


def calculate_profits(X, beta):
    """
    Calculate deterministic profits V_{i,m} for all firms in all markets.
    
    In the "no heter" model, all firms have the same profit function:
    V_{i,m} = X_m' * beta
    
    Args:
        X: Market fundamentals (n_markets x k)
        beta: Coefficients (k,)
    
    Returns:
        V: Profit matrix (n_markets x 6) - same for all firms
    """
    V_base = X @ beta  # n_markets
    # All 6 firms have the same deterministic profit in "no heter" model
    V = np.tile(V_base[:, np.newaxis], (1, 6))  # n_markets x 6
    return V


def calculate_entry_probabilities_profit_rule(V, delta):
    """
    Calculate probabilities using the "profit" rule from R code.
    
    This matches lines 137-151 in entry_mle_simplified.R
    Firms are ranked by profit, highest moves first.
    
    Args:
        V: Deterministic profits (n_markets x 6)
        delta: Competition effect parameter
    
    Returns:
        p0, p1, p2plus: Probabilities for each outcome
    """
    n_markets, n_firms = V.shape
    
    p0 = np.zeros(n_markets)
    p1 = np.zeros(n_markets)
    
    for m in range(n_markets):
        # P(N=0): All firms stay out
        # Product of Φ(-V_i) for all firms
        p0[m] = np.prod(norm.cdf(-V[m, :]))
        
        # P(N=1): Exactly one firm enters
        # Sort profits in descending order
        V_sorted = np.sort(V[m, :])[::-1]  # Descending
        V1 = V_sorted[0]  # Highest profit
        V2 = V_sorted[1]  # Second highest profit
        
        # R formula (line 148):
        # p1.an = pnorm(list[1])*pnorm(-list[1] + delta) - 
        #         (pnorm(-list[1]+delta) - pnorm(-list[1]))*
        #         (pnorm(-list[2]+delta) - pnorm(-list[2]))*
        #         (1 - pnorm((list[2]-list[1])/2))
        
        term1 = norm.cdf(V1) * norm.cdf(-V1 + delta)
        term2 = (norm.cdf(-V1 + delta) - norm.cdf(-V1))
        term3 = (norm.cdf(-V2 + delta) - norm.cdf(-V2))
        term4 = (1 - norm.cdf((V2 - V1) / 2))
        
        p1[m] = term1 - term2 * term3 * term4
    
    # P(N>=2)
    p2plus = 1.0 - p0 - p1
    
    # Ensure non-negative and normalize
    p0 = np.maximum(p0, 0)
    p1 = np.maximum(p1, 0)
    p2plus = np.maximum(p2plus, 0)
    
    total = p0 + p1 + p2plus
    p0 = p0 / total
    p1 = p1 / total
    p2plus = p2plus / total
    
    return p0, p1, p2plus


def negative_log_likelihood(theta, X, outcome_0, outcome_1, outcome_2plus):
    """
    Negative log-likelihood function matching R implementation
    """
    try:
        k = X.shape[1]
        beta = theta[:k]
        delta = np.exp(theta[k])  # Ensure positive
        
        # Calculate profits
        V = calculate_profits(X, beta)
        
        # Calculate probabilities using profit rule
        p0, p1, p2plus = calculate_entry_probabilities_profit_rule(V, delta)
        
        # Clip probabilities to avoid log(0)
        eps = 1e-10
        p0 = np.clip(p0, eps, 1-eps)
        p1 = np.clip(p1, eps, 1-eps)
        p2plus = np.clip(p2plus, eps, 1-eps)

        # Log-likelihood
        ll = np.sum(outcome_0 * np.log(p0) + outcome_1 * np.log(p1) + outcome_2plus * np.log(p2plus))

        if np.isnan(ll) or np.isinf(ll):
            return 1e10

        return -ll
    except Exception as e:
        return 1e10


def estimate_model(data_path='Berry_data.csv', verbose=True):
    """
    Estimate the Berry airline entry model with corrected specification
    """
    if verbose:
        print("="*70)
        print(" "*15 + "BERRY AIRLINE ENTRY MODEL - CORRECTED")
        print(" "*10 + "Matching R Reference Implementation")
        print("="*70)

    # Load data
    if verbose:
        print("\nLoading and preprocessing data...")
    X, entry, outcome_0, outcome_1, outcome_2plus = load_and_preprocess_data(data_path)

    n_markets = X.shape[0]
    n_params = X.shape[1]

    if verbose:
        print(f"Markets: {n_markets}")
        print(f"Parameters: {n_params}")
        print(f"Covariates: intercept, log(population), log(distance), log(distance^2)")
        print(f"Outcome distribution: N=0: {outcome_0.sum():.0f}, N=1: {outcome_1.sum():.0f}, N>=2: {outcome_2plus.sum():.0f}")

    # Starting values
    theta0 = np.zeros(n_params + 1)
    theta0[0] = 0.0  # Intercept
    theta0[-1] = 0.0  # log(delta) = 0 => delta = 1

    if verbose:
        ll0 = -negative_log_likelihood(theta0, X, outcome_0, outcome_1, outcome_2plus)
        print(f"\nStarting log-likelihood: {ll0:.4f}")
        print("\nOptimizing with BFGS (matching R)...")

    # Optimize using BFGS to match R
    result = minimize(
        negative_log_likelihood,
        theta0,
        args=(X, outcome_0, outcome_1, outcome_2plus),
        method='BFGS',
        options={'maxiter': 1000, 'disp': verbose}
    )

    # Extract results
    theta_hat = result.x
    beta_hat = theta_hat[:n_params]
    delta_hat = np.exp(theta_hat[n_params])
    ll_final = -result.fun

    # Calculate predictions
    V = calculate_profits(X, beta_hat)
    p0, p1, p2plus = calculate_entry_probabilities_profit_rule(V, delta_hat)

    predicted = np.argmax(np.column_stack([p0, p1, p2plus]), axis=1)
    actual = np.argmax(np.column_stack([outcome_0, outcome_1, outcome_2plus]), axis=1)

    overall_accuracy = np.mean(predicted == actual)

    # Display results
    if verbose:
        print(f"\nOptimization complete: {result.success}")
        print(f"Message: {result.message}")

        print("\n" + "="*70)
        print("ESTIMATION RESULTS - CORRECTED MODEL")
        print("="*70)

        param_names = ['Constant (α)', 'Log(Population) (β₁)',
                       'Log(Distance) (β₂)', 'Log(Distance²) (β₃)']

        print("\nParameter Estimates:")
        print("-" * 70)
        print(f"{'Parameter':<35} {'Estimate':>15}")
        print("-" * 70)
        for i, name in enumerate(param_names):
            print(f"{name:<35} {beta_hat[i]:>15.6f}")
        print(f"{'Competition Effect (δ)':<35} {delta_hat:>15.6f}")
        print("-" * 70)

        print(f"\nLog-Likelihood: {ll_final:.4f}")
        print(f"Number of Observations: {n_markets}")
        print(f"Overall Prediction Accuracy: {100*overall_accuracy:.2f}%")

        print("\n" + "="*70)
        print("COMPARISON WITH PREVIOUS MODEL")
        print("="*70)
        print("Previous model used:")
        print("  - Incorrect P(N=1) formula")
        print("  - Different covariates (avg log pop, tourism, passengers)")
        print("  - Airline-specific size parameters")
        print("\nCorrected model uses:")
        print("  - Correct P(N=1) formula from R code")
        print("  - R specification: log(pop), log(dist), log(dist²)")
        print("  - No airline heterogeneity (all firms same)")
        print("="*70)

    # Return results
    results = {
        'beta': beta_hat,
        'delta': delta_hat,
        'log_likelihood': ll_final,
        'n_obs': n_markets,
        'theta': theta_hat,
        'success': result.success,
        'accuracy': overall_accuracy,
        'predictions': predicted,
        'actual': actual,
        'hessian': result.hess_inv if hasattr(result, 'hess_inv') else None
    }

    return results


if __name__ == "__main__":
    import sys

    # Redirect output to file for WSL2
    sys.stdout = open('berry_corrected_output.txt', 'w')
    sys.stderr = sys.stdout

    # Run estimation
    results = estimate_model()

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print("\nKey Changes Made:")
    print("1. ✓ Fixed P(N=1) calculation to match R formula")
    print("2. ✓ Changed covariates to: log(pop), log(dist), log(dist²)")
    print("3. ✓ Removed airline-specific size parameters")
    print("4. ✓ Used BFGS optimizer (matching R)")
    print("\nFinal Results:")
    print(f"  - Log-Likelihood: {results['log_likelihood']:.4f}")
    print(f"  - Competition Effect (δ): {results['delta']:.4f}")
    print(f"  - Prediction Accuracy: {100*results['accuracy']:.2f}%")
    print("="*70 + "\n")

    sys.stdout.close()

    # Print to console
    sys.stdout = sys.__stdout__
    print("Corrected estimation complete. Results saved to berry_corrected_output.txt")
