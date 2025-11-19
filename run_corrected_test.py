"""Test corrected model with file output"""
import sys
import traceback

# Open output file
outfile = open('corrected_test_output.txt', 'w')
sys.stdout = outfile
sys.stderr = outfile

try:
    print("Importing modules...")
    from berry_corrected import estimate_model
    
    print("Starting estimation...")
    results = estimate_model(verbose=True)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Success: {results['success']}")
    print(f"Log-Likelihood: {results['log_likelihood']:.4f}")
    print(f"Delta: {results['delta']:.6f}")
    print(f"Beta: {results['beta']}")
    print("="*70)
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

outfile.close()
print("Done - check corrected_test_output.txt", file=sys.__stdout__)

