"""Test the corrected Berry model"""

from berry_corrected import estimate_model
import sys

# Run estimation
print("Starting corrected Berry model estimation...")
results = estimate_model(verbose=True)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"Log-Likelihood: {results['log_likelihood']:.4f}")
print(f"Delta: {results['delta']:.4f}")
print(f"Beta: {results['beta']}")
print(f"Success: {results['success']}")
print("="*70)

