"""
Berry Airline Entry Model - Maximum Likelihood Estimation
Implementation of Berry (1992) static entry game for airline industry
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class BerryEntryModel:
    """
    Implements the Berry (1992) airline entry model using MLE.
    
    Model: π_{i,m} = V_{i,m} - δ * N_{-i,m} + ε_{i,m}
    where V_{i,m} = X_m * β + size_i
    """
    
    def __init__(self, data_path='Berry_data.csv'):
        """Initialize and load data."""
        self.data = pd.read_csv(data_path)
        self.airlines = ['airlineaa', 'airlinedl', 'airlineua', 
                        'airlineal', 'airlinelcc', 'airlinewn']
        # Airline sizes as specified in instructions
        self.airline_sizes = {
            'airlineaa': 1000,
            'airlinedl': 1500,
            'airlineua': 1000,
            'airlineal': 1500,
            'airlinelcc': 500,
            'airlinewn': 500
        }
        self.preprocess_data()
        
    def preprocess_data(self):
        """Preprocess data: create market fundamentals and outcome variables."""
        print("Preprocessing data...")

        # Extract entry decisions
        self.entry_decisions = self.data[self.airlines].values  # M x 6 matrix
        self.n_markets = len(self.data)
        self.n_firms = len(self.airlines)

        # Calculate number of entrants per market
        self.n_entrants = self.entry_decisions.sum(axis=1)

        # Create outcome indicators
        self.outcome_0 = (self.n_entrants == 0).astype(int)  # No entry
        self.outcome_1 = (self.n_entrants == 1).astype(int)  # Monopoly
        self.outcome_2plus = (self.n_entrants >= 2).astype(int)  # Competition

        # Transform market variables (log transformations and standardization)
        self.data['log_pop1'] = np.log(self.data['population1'])
        self.data['log_pop2'] = np.log(self.data['population2'])
        self.data['log_distance'] = np.log(self.data['distance'])
        self.data['log_passengers'] = np.log(self.data['passengers'])

        # Standardize variables for numerical stability
        self.data['log_pop1_std'] = (self.data['log_pop1'] - self.data['log_pop1'].mean()) / self.data['log_pop1'].std()
        self.data['log_pop2_std'] = (self.data['log_pop2'] - self.data['log_pop2'].mean()) / self.data['log_pop2'].std()
        self.data['log_distance_std'] = (self.data['log_distance'] - self.data['log_distance'].mean()) / self.data['log_distance'].std()
        self.data['log_passengers_std'] = (self.data['log_passengers'] - self.data['log_passengers'].mean()) / self.data['log_passengers'].std()

        # Construct market fundamentals matrix X (standardized)
        # Columns: intercept, log_pop1, log_pop2, tourism1, tourism2, log_distance, log_passengers
        self.X = np.column_stack([
            np.ones(self.n_markets),  # Intercept
            self.data['log_pop1_std'],
            self.data['log_pop2_std'],
            self.data['tourism1'],
            self.data['tourism2'],
            self.data['log_distance_std'],
            self.data['log_passengers_std']
        ])

        self.n_params_beta = self.X.shape[1]

        print(f"Data loaded: {self.n_markets} markets, {self.n_firms} airlines")
        print(f"Outcome distribution: N=0: {self.outcome_0.sum()}, "
              f"N=1: {self.outcome_1.sum()}, N>=2: {self.outcome_2plus.sum()}")
    
    def calculate_deterministic_profits(self, beta):
        """
        Calculate deterministic profit V_{i,m} for each airline-market pair.

        Args:
            beta: Coefficient vector for market fundamentals (length = n_params_beta)

        Returns:
            V: Matrix of deterministic profits (n_markets x n_firms)
        """
        # Base profit from market fundamentals: X * beta
        base_profit = self.X @ beta  # n_markets x 1

        # Add airline-specific sizes (scaled down for numerical stability)
        V = np.zeros((self.n_markets, self.n_firms))
        for i, airline in enumerate(self.airlines):
            # Scale airline sizes by 1/1000 for numerical stability
            V[:, i] = base_profit + self.airline_sizes[airline] / 1000.0

        return V
    
    def calculate_probabilities(self, V, delta):
        """
        Calculate entry probabilities for each market.

        Args:
            V: Deterministic profits (n_markets x n_firms)
            delta: Competitive effect parameter

        Returns:
            p0, p1, p2plus: Probabilities for each outcome
        """
        n_markets = V.shape[0]
        p0 = np.zeros(n_markets)
        p1 = np.zeros(n_markets)
        p2plus = np.zeros(n_markets)

        # Clip V values to prevent extreme probabilities
        V = np.clip(V, -10, 10)
        delta = np.clip(delta, 0.01, 10)

        for m in range(n_markets):
            V_m = V[m, :]  # Profits for all firms in market m

            # P(N=0): All firms have negative profit
            # Use log-sum-exp trick for numerical stability
            log_p0 = np.sum(norm.logcdf(-V_m))
            p0[m] = np.exp(log_p0)

            # P(N=1): Sequential entry - highest profit firm enters, others stay out
            # Find firm with highest deterministic profit
            max_idx = np.argmax(V_m)
            V_max = V_m[max_idx]

            # Probability highest firm enters
            prob_max_enters = norm.cdf(V_max)

            # Probability all others stay out given max firm entered
            log_prob_others_out = 0.0
            for j in range(self.n_firms):
                if j != max_idx:
                    # Other firms stay out if V_j - delta < 0
                    log_prob_others_out += norm.logcdf(-(V_m[j] - delta))

            prob_others_out = np.exp(log_prob_others_out)
            p1[m] = prob_max_enters * prob_others_out

            # P(N>=2): Residual probability
            p2plus[m] = 1.0 - p0[m] - p1[m]

            # Ensure probabilities are valid
            if p2plus[m] < 0:
                # Renormalize
                total = p0[m] + p1[m]
                if total > 0:
                    p0[m] = p0[m] / total
                    p1[m] = p1[m] / total
                    p2plus[m] = 0.0
                else:
                    p0[m] = 1.0/3
                    p1[m] = 1.0/3
                    p2plus[m] = 1.0/3

        return p0, p1, p2plus

    def negative_log_likelihood(self, theta):
        """
        Calculate negative log-likelihood for optimization.

        Args:
            theta: Parameter vector [beta (n_params_beta), delta]

        Returns:
            Negative log-likelihood value
        """
        try:
            # Extract parameters
            beta = theta[:self.n_params_beta]
            delta = np.exp(theta[self.n_params_beta])  # Exponentiate to ensure delta > 0

            # Calculate deterministic profits
            V = self.calculate_deterministic_profits(beta)

            # Calculate probabilities
            p0, p1, p2plus = self.calculate_probabilities(V, delta)

            # Ensure probabilities are positive (numerical stability)
            eps = 1e-10
            p0 = np.clip(p0, eps, 1-eps)
            p1 = np.clip(p1, eps, 1-eps)
            p2plus = np.clip(p2plus, eps, 1-eps)

            # Calculate log-likelihood
            log_lik = (self.outcome_0 * np.log(p0) +
                       self.outcome_1 * np.log(p1) +
                       self.outcome_2plus * np.log(p2plus))

            total_log_lik = np.sum(log_lik)

            # Check for NaN or Inf
            if np.isnan(total_log_lik) or np.isinf(total_log_lik):
                return 1e10

            # Return negative for minimization
            return -total_log_lik

        except Exception as e:
            # Return large value if calculation fails
            return 1e10

    def estimate(self, starting_values=None, method='Nelder-Mead', verbose=True):
        """
        Estimate model parameters using MLE.

        Args:
            starting_values: Initial parameter values (optional)
            method: Optimization method ('BFGS', 'Nelder-Mead', etc.)
            verbose: Print optimization progress

        Returns:
            results: Dictionary with parameter estimates and statistics
        """
        print("\n" + "="*60)
        print("BERRY AIRLINE ENTRY MODEL - MLE ESTIMATION")
        print("="*60)

        # Set starting values
        if starting_values is None:
            # Start with small values for beta, small positive for log(delta)
            starting_values = np.zeros(self.n_params_beta + 1)
            starting_values[0] = 1.0  # Intercept
            starting_values[-1] = -1.0  # log(delta) ≈ 0.37

        print(f"\nOptimizing with {len(starting_values)} parameters...")
        print(f"Method: {method}")
        print(f"Starting log-likelihood: {-self.negative_log_likelihood(starting_values):.4f}")

        # Optimize with Nelder-Mead first (more robust)
        result = minimize(
            self.negative_log_likelihood,
            starting_values,
            method=method,
            options={'disp': verbose, 'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
        )

        if not result.success:
            print("\nWARNING: Optimization did not converge!")
            print(f"Message: {result.message}")
        else:
            print("\nOptimization successful!")

        # Extract results
        theta_hat = result.x
        beta_hat = theta_hat[:self.n_params_beta]
        delta_hat = np.exp(theta_hat[self.n_params_beta])

        # Calculate standard errors from Hessian
        try:
            # Re-optimize with Hessian
            result_hess = minimize(
                self.negative_log_likelihood,
                theta_hat,
                method='BFGS',
                options={'disp': False}
            )

            # Numerical Hessian
            from scipy.optimize import approx_fprime
            hessian = np.zeros((len(theta_hat), len(theta_hat)))
            eps = 1e-5

            for i in range(len(theta_hat)):
                def grad_i(x):
                    return approx_fprime(x, self.negative_log_likelihood, eps)[i]
                hessian[i, :] = approx_fprime(theta_hat, grad_i, eps)

            # Standard errors from inverse Hessian
            cov_matrix = np.linalg.inv(hessian)
            se = np.sqrt(np.diag(cov_matrix))

            # Adjust SE for delta (delta transformation)
            se_beta = se[:self.n_params_beta]
            se_delta = se[self.n_params_beta] * delta_hat  # Delta method

        except:
            print("\nWarning: Could not compute standard errors")
            se_beta = np.full(self.n_params_beta, np.nan)
            se_delta = np.nan

        # Store results
        self.results = {
            'beta': beta_hat,
            'delta': delta_hat,
            'se_beta': se_beta,
            'se_delta': se_delta,
            'log_likelihood': -result.fun,
            'n_obs': self.n_markets,
            'convergence': result.success,
            'theta': theta_hat
        }

        return self.results

    def display_results(self):
        """Display estimation results in a formatted table."""
        if not hasattr(self, 'results'):
            print("No results to display. Run estimate() first.")
            return

        print("\n" + "="*60)
        print("ESTIMATION RESULTS")
        print("="*60)

        # Parameter names
        param_names = [
            'Constant (α)',
            'Log Population Origin (β₁)',
            'Log Population Dest (β₂)',
            'Tourism Origin (β₃)',
            'Tourism Dest (β₄)',
            'Log Distance (β₅)',
            'Log Passengers (β₆)',
            'Competition Effect (δ)'
        ]

        # Combine parameters
        all_params = np.append(self.results['beta'], self.results['delta'])
        all_se = np.append(self.results['se_beta'], self.results['se_delta'])

        # Calculate t-statistics
        t_stats = all_params / all_se

        print("\nParameter Estimates:")
        print("-" * 60)
        print(f"{'Parameter':<30} {'Estimate':>12} {'Std Error':>12} {'t-stat':>10}")
        print("-" * 60)

        for i, name in enumerate(param_names):
            print(f"{name:<30} {all_params[i]:>12.6f} {all_se[i]:>12.6f} {t_stats[i]:>10.3f}")

        print("-" * 60)
        print(f"\nLog-Likelihood: {self.results['log_likelihood']:.4f}")
        print(f"Number of Observations: {self.results['n_obs']}")
        print(f"Number of Parameters: {len(all_params)}")

        # Highlight the 3 key parameters
        print("\n" + "="*60)
        print("KEY PARAMETERS (as requested)")
        print("="*60)
        print(f"1. Market Size Effect (β₁):     {self.results['beta'][1]:>10.6f}")
        print(f"2. Distance Effect (β₅):         {self.results['beta'][5]:>10.6f}")
        print(f"3. Competition Effect (δ):       {self.results['delta']:>10.6f}")
        print("="*60)

    def predict(self):
        """Generate predictions for entry outcomes."""
        if not hasattr(self, 'results'):
            print("No results available. Run estimate() first.")
            return

        # Calculate predicted probabilities
        V = self.calculate_deterministic_profits(self.results['beta'])
        p0, p1, p2plus = self.calculate_probabilities(V, self.results['delta'])

        # Predicted outcomes (most likely outcome)
        predicted_n = np.zeros(self.n_markets)
        for m in range(self.n_markets):
            probs = [p0[m], p1[m], p2plus[m]]
            predicted_n[m] = np.argmax(probs)  # 0, 1, or 2

        # Actual outcomes
        actual_n = np.zeros(self.n_markets)
        actual_n[self.outcome_1 == 1] = 1
        actual_n[self.outcome_2plus == 1] = 2

        # Calculate prediction accuracy
        correct_0 = np.sum((predicted_n == 0) & (actual_n == 0))
        correct_1 = np.sum((predicted_n == 1) & (actual_n == 1))
        correct_2 = np.sum((predicted_n == 2) & (actual_n == 2))

        total_correct = correct_0 + correct_1 + correct_2
        accuracy = total_correct / self.n_markets

        print("\n" + "="*60)
        print("PREDICTION ACCURACY")
        print("="*60)
        print(f"Correctly predicted N=0:    {correct_0}/{self.outcome_0.sum()} "
              f"({100*correct_0/max(self.outcome_0.sum(),1):.1f}%)")
        print(f"Correctly predicted N=1:    {correct_1}/{self.outcome_1.sum()} "
              f"({100*correct_1/max(self.outcome_1.sum(),1):.1f}%)")
        print(f"Correctly predicted N>=2:   {correct_2}/{self.outcome_2plus.sum()} "
              f"({100*correct_2/max(self.outcome_2plus.sum(),1):.1f}%)")
        print(f"\nOverall Accuracy: {100*accuracy:.2f}%")
        print("="*60)

        return predicted_n, p0, p1, p2plus


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" "*15 + "BERRY AIRLINE ENTRY MODEL")
    print(" "*10 + "Static Entry Game Estimation - Berry (1992)")
    print("="*70)

    # Initialize model
    model = BerryEntryModel('Berry_data.csv')

    # Estimate parameters
    results = model.estimate(method='BFGS', verbose=False)

    # Display results
    model.display_results()

    # Generate predictions
    model.predict()

    print("\n" + "="*70)
    print("Estimation complete!")
    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    model = main()
