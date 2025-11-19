# Berry Airline Entry Model - Python Implementation

## Overview

This project implements the static entry game model from Berry (1992) to analyze airline entry decisions across 2,742 U.S. city-pair markets. The implementation estimates how market fundamentals and competitive effects influence airlines' decisions to enter or stay out of specific routes.

## Files

### Main Implementation
- **`berry_entry_model_final.py`** - Complete, production-ready implementation
  - Clean, well-documented code
  - Handles missing data
  - Numerical stability measures
  - Comprehensive results output

### Results and Documentation
- **`FINAL_RESULTS.txt`** - Detailed estimation results and interpretation
- **`berry_writeup.tex`** - 2-page LaTeX writeup (compile with `pdflatex`)

### Supporting Files
- **`berry_model_v2.py`** - Development version with debugging features
- **`run_estimation.py`** - Standalone estimation script
- **`final_estimation.py`** - Extended results with predictions
- **`estimation_results.txt`** - Initial results file

## Quick Start

### Run the Model

```python
python3 berry_entry_model_final.py
```

### Use as a Module

```python
from berry_entry_model_final import estimate_model

# Run estimation
results = estimate_model()

# Access results
print(f"Competition Effect: {results['delta']:.4f}")
print(f"Log-Likelihood: {results['log_likelihood']:.4f}")
print(f"Accuracy: {100*results['accuracy']:.2f}%")
```

## Model Specification

### Profit Function
```
π_{i,m} = V_{i,m} - δ * N_{-i,m} + ε_{i,m}
```

where:
- `V_{i,m}` = Deterministic profit (function of market characteristics)
- `δ` = Competition effect (profit reduction per competitor)
- `N_{-i,m}` = Number of other firms entering
- `ε_{i,m}` ~ N(0,1) = Idiosyncratic shock

### Deterministic Profit
```
V_{i,m} = α + β₁*MarketSize + β₂*Tourism + β₃*Distance + β₄*Passengers + AirlineSize_i
```

## Key Results

### Parameter Estimates

| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| α (Constant) | 0.3293 | Base profitability |
| β₁ (Market Size) | -0.1258 | Negative after controlling for passengers |
| β₂ (Tourism) | -0.0873 | Slight negative effect |
| β₃ (Distance) | **0.2868** | Longer routes more profitable |
| β₄ (Passengers) | **0.8397** | Strongest positive effect |
| δ (Competition) | **1.4809** | Substantial competitive effect |

### Model Fit
- **Log-Likelihood**: -1908.39
- **Convergence**: Successful (L-BFGS-B)
- **Markets**: 2,742
- **Parameters**: 6

### Data Distribution
- **N=0** (No Entry): 200 markets (7.3%)
- **N=1** (Monopoly): 840 markets (30.6%)
- **N≥2** (Competition): 1,702 markets (62.1%)

## Economic Insights

1. **Passenger volume is the primary driver** of airline entry (β₄ = 0.84)

2. **Competition significantly reduces profitability** (δ = 1.48), explaining why many markets remain monopolies

3. **Longer routes are more profitable** (β₃ = 0.29), consistent with airline pricing strategies

4. **Market structure is endogenous** - the high competition effect creates natural entry barriers

## Technical Features

✅ **Data Handling**
- Automatic handling of missing values (3 NaN in passengers column)
- Data standardization for numerical stability

✅ **Numerical Stability**
- Value clipping to prevent overflow
- Log-sum-exp tricks for probability calculations
- Robust error handling

✅ **Optimization**
- L-BFGS-B algorithm for fast convergence
- Proper parameter transformations (log(δ) to ensure positivity)
- Convergence diagnostics

✅ **Code Quality**
- Clean, modular design
- Comprehensive documentation
- Type hints and docstrings
- Production-ready

## Requirements

```
numpy
pandas
scipy
```

Install with:
```bash
pip install numpy pandas scipy
```

Or with conda:
```bash
conda install numpy pandas scipy
```

## Compilation of LaTeX Document

To compile the writeup to PDF:

```bash
pdflatex berry_writeup.tex
pdflatex berry_writeup.tex  # Run twice for references
```

## References

Berry, S. T. (1992). "Estimation of a Model of Entry in the Airline Industry." *Econometrica*, 60(4), 889-917.

## Author

Vatsal Mitesh Tailor - 2025

## License

This implementation is for educational and research purposes.
