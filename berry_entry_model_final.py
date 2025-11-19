"""
Berry Airline Entry Model - Final Implementation
Static Entry Game Estimation following Berry (1992)

This module implements maximum likelihood estimation of a static entry game
for airline markets. The model estimates how market fundamentals and competitive
effects influence airlines' entry decisions.

Author: Vatsal Mitesh Tailor
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_path='Berry_data.csv'):
    """
    Load and preprocess the airline entry data.
    
    Returns:
        X: Market fundamentals matrix (n_markets x k)
        airline_sizes: Airline size parameters (6,)
        entry: Entry decisions matrix (n_markets x 6)
        outcome_0, outcome_1, outcome_2plus: Outcome indicators
    """
    data = pd.read_csv(data_path)
    
    # Handle missing values in passengers column (3 NaN values)
    data['passengers'] = data['passengers'].fillna(data['passengers'].median())
    
    # Airline names and sizes (scaled)
    airlines = ['airlineaa', 'airlinedl', 'airlineua', 'airlineal', 'airlinelcc', 'airlinewn']
    airline_sizes = np.array([0.1, 0.15, 0.1, 0.15, 0.05, 0.05])  # AA, DL, UA, AL, LCC, WN
    
    # Entry decisions
    entry = data[airlines].values  # 2742 x 6
    n_entrants = entry.sum(axis=1)
    
    # Outcome indicators
    outcome_0 = (n_entrants == 0).astype(float)
    outcome_1 = (n_entrants == 1).astype(float)
    outcome_2plus = (n_entrants >= 2).astype(float)
    
    # Market fundamentals (standardized for numerical stability)
    pop_avg = (np.log(data['population1']) + np.log(data['population2'])) / 2
    pop_avg_std = (pop_avg - pop_avg.mean()) / pop_avg.std()
    
    tourism_total = data['tourism1'] + data['tourism2']
    
    dist_std = (np.log(data['distance']) - np.log(data['distance']).mean()) / np.log(data['distance']).std()
    pass_std = (np.log(data['passengers']) - np.log(data['passengers']).mean()) / np.log(data['passengers']).std()
    
    # Construct X matrix: intercept, pop, tourism, distance, passengers
    X = np.column_stack([
        np.ones(len(data)),
        pop_avg_std,
        tourism_total,
        dist_std,
        pass_std
    ])
    
    return X, airline_sizes, entry, outcome_0, outcome_1, outcome_2plus


def calculate_profits(X, airline_sizes, beta):
    """
    Calculate deterministic profits V_{i,m} for all firms in all markets.
    
    V_{i,m} = X_m' * beta + airline_size_i
    
    Args:
        X: Market fundamentals (n_markets x k)
        airline_sizes: Airline size parameters (6,)
        beta: Coefficients (k,)
    
    Returns:
        V: Profit matrix (n_markets x 6)
    """
    base_profit = X @ beta  # n_markets
    V = base_profit[:, np.newaxis] + airline_sizes[np.newaxis, :]  # Broadcasting
    return V


def calculate_entry_probabilities(V, delta):
    """
    Calculate probabilities of 0, 1, or 2+ firms entering each market.
    
    Uses Berry (1992) sequential entry assumption:
    - Firms with higher expected profits move first
    - Each firm considers the number of competitors when deciding
    
    Args:
        V: Deterministic profits (n_markets x 6)
        delta: Competition effect parameter
    
    Returns:
        p0, p1, p2plus: Probabilities for each outcome
    """
    n_markets, n_firms = V.shape
    
    # Clip to avoid extreme values
    V = np.clip(V, -5, 5)
    delta = np.clip(delta, 0.01, 5)
    
    p0 = np.zeros(n_markets)
    p1 = np.zeros(n_markets)
    
    for m in range(n_markets):
        # P(N=0): All firms stay out
        p0[m] = np.exp(np.sum(norm.logcdf(-V[m, :])))
        
        # P(N=1): Exactly one firm enters
        # Firm with highest V enters, others stay out given entry
        max_idx = np.argmax(V[m, :])
        V_max = V[m, max_idx]
        
        # Prob max firm enters AND all others stay out given entry
        log_p1 = norm.logcdf(V_max)
        for j in range(n_firms):
            if j != max_idx:
                log_p1 += norm.logcdf(-(V[m, j] - delta))
        
        p1[m] = np.exp(log_p1)
    
    # P(N>=2)
    p2plus = 1.0 - p0 - p1
    p2plus = np.maximum(p2plus, 0)  # Ensure non-negative
    
    # Renormalize to ensure probabilities sum to 1
    total = p0 + p1 + p2plus
    p0 = p0 / total
    p1 = p1 / total
    p2plus = p2plus / total
    
    return p0, p1, p2plus


def negative_log_likelihood(theta, X, airline_sizes, outcome_0, outcome_1, outcome_2plus):
    """
    Negative log-likelihood function for MLE optimization.
    
    Args:
        theta: Parameters [beta (k), log(delta)]
        X, airline_sizes, outcome_0, outcome_1, outcome_2plus: Data
    
    Returns:
        Negative log-likelihood value
    """
    try:
        k = X.shape[1]
        beta = theta[:k]
        delta = np.exp(theta[k])  # Ensure positive
        
        # Calculate profits
        V = calculate_profits(X, airline_sizes, beta)
        
        # Calculate probabilities
        p0, p1, p2plus = calculate_entry_probabilities(V, delta)
        
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
    except:
        return 1e10


def estimate_model(data_path='Berry_data.csv', verbose=True):
    """
    Estimate the Berry airline entry model.

    Args:
        data_path: Path to data file
        verbose: Whether to print results

    Returns:
        Dictionary with estimation results
    """
    if verbose:
        print("="*70)
        print(" "*20 + "BERRY AIRLINE ENTRY MODEL")
        print(" "*15 + "Static Entry Game Estimation - Berry (1992)")
        print("="*70)

    # Load data
    if verbose:
        print("\nLoading and preprocessing data...")
    X, airline_sizes, entry, outcome_0, outcome_1, outcome_2plus = load_and_preprocess_data(data_path)

    n_markets = X.shape[0]
    n_params = X.shape[1]

    if verbose:
        print(f"Markets: {n_markets}")
        print(f"Parameters: {n_params}")
        print(f"Outcome distribution: N=0: {outcome_0.sum():.0f}, N=1: {outcome_1.sum():.0f}, N>=2: {outcome_2plus.sum():.0f}")

    # Starting values
    theta0 = np.zeros(n_params + 1)
    theta0[0] = 0.0  # Intercept
    theta0[-1] = -2.0  # log(delta) ≈ 0.135

    if verbose:
        ll0 = -negative_log_likelihood(theta0, X, airline_sizes, outcome_0, outcome_1, outcome_2plus)
        print(f"\nStarting log-likelihood: {ll0:.4f}")
        print("\nOptimizing with L-BFGS-B...")

    # Optimize
    result = minimize(
        negative_log_likelihood,
        theta0,
        args=(X, airline_sizes, outcome_0, outcome_1, outcome_2plus),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'disp': False}
    )

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

    # Display results
    if verbose:
        print(f"\nOptimization complete: {result.success}")
        print(f"Message: {result.message}")

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
        print(f"Overall Prediction Accuracy: {100*overall_accuracy:.2f}%")

        print("\n" + "="*70)
        print("KEY PARAMETERS")
        print("="*70)
        print(f"1. Market Size Effect (β₁):      {beta_hat[1]:>12.6f}")
        print(f"2. Distance Effect (β₃):          {beta_hat[3]:>12.6f}")
        print(f"3. Competition Effect (δ):        {delta_hat:>12.6f}")
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
        'actual': actual
    }

    return results


if __name__ == "__main__":
    # Run estimation
    results = estimate_model()

    print("\n" + "="*70)
    print("ESTIMATION COMPLETE!")
    print("="*70)
    print("\nKey Results:")
    print(f"  - Log-Likelihood: {results['log_likelihood']:.4f}")
    print(f"  - Competition Effect (δ): {results['delta']:.4f}")
    print(f"  - Prediction Accuracy: {100*results['accuracy']:.2f}%")
    print("\nSee FINAL_RESULTS.txt for detailed interpretation.")
    print("="*70 + "\n")
