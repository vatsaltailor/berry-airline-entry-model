"""
Quick test of the final Berry model implementation
"""

import sys
sys.stdout = open('test_output.txt', 'w')
sys.stderr = sys.stdout

print("="*70)
print("TESTING BERRY ENTRY MODEL FINAL IMPLEMENTATION")
print("="*70)

try:
    from berry_entry_model_final import estimate_model
    print("\n✓ Successfully imported berry_entry_model_final")
    
    print("\nRunning estimation...")
    results = estimate_model(verbose=False)
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    print(f"\n✓ Optimization Success: {results['success']}")
    print(f"✓ Log-Likelihood: {results['log_likelihood']:.4f}")
    print(f"✓ Number of Observations: {results['n_obs']}")
    
    print("\nParameter Estimates:")
    print(f"  β₁ (Market Size):    {results['beta'][1]:>10.6f}")
    print(f"  β₃ (Distance):       {results['beta'][3]:>10.6f}")
    print(f"  β₄ (Passengers):     {results['beta'][4]:>10.6f}")
    print(f"  δ (Competition):     {results['delta']:>10.6f}")
    
    print(f"\n✓ Prediction Accuracy: {100*results['accuracy']:.2f}%")
    
    # Validation checks
    print("\n" + "="*70)
    print("VALIDATION CHECKS")
    print("="*70)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Optimization converged
    if results['success']:
        print("✓ Check 1: Optimization converged successfully")
        checks_passed += 1
    else:
        print("✗ Check 1: Optimization did not converge")
    
    # Check 2: Log-likelihood is finite
    if -3000 < results['log_likelihood'] < -1000:
        print("✓ Check 2: Log-likelihood is in reasonable range")
        checks_passed += 1
    else:
        print("✗ Check 2: Log-likelihood out of expected range")
    
    # Check 3: Delta is positive
    if results['delta'] > 0:
        print("✓ Check 3: Competition effect (δ) is positive")
        checks_passed += 1
    else:
        print("✗ Check 3: Competition effect is not positive")
    
    # Check 4: Passengers coefficient is positive
    if results['beta'][4] > 0:
        print("✓ Check 4: Passengers coefficient is positive (as expected)")
        checks_passed += 1
    else:
        print("✗ Check 4: Passengers coefficient is not positive")
    
    # Check 5: Distance coefficient is positive
    if results['beta'][3] > 0:
        print("✓ Check 5: Distance coefficient is positive (as expected)")
        checks_passed += 1
    else:
        print("✗ Check 5: Distance coefficient is not positive")
    
    print("\n" + "="*70)
    print(f"VALIDATION SUMMARY: {checks_passed}/{total_checks} checks passed")
    print("="*70)
    
    if checks_passed == total_checks:
        print("\n✅ ALL TESTS PASSED - IMPLEMENTATION IS WORKING CORRECTLY")
    else:
        print(f"\n⚠ WARNING: {total_checks - checks_passed} checks failed")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n✗ TEST FAILED")

sys.stdout.close()

# Print to console
sys.stdout = sys.__stdout__
print("Test complete. Results saved to test_output.txt")

