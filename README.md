# Berry Airline Entry Model

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vatsaltailor/berry-airline-entry-model)

Python implementation of the Berry (1992) static entry game for U.S. airline markets. Estimates how market characteristics and competition affect airline entry decisions using maximum likelihood.

## Overview

This repository implements a discrete-choice entry model for 2,742 origin-destination airline markets. Each market has up to six potential carriers (American, Delta, United, Alaska, Low-Cost Carriers, Southwest) who decide whether to operate a route based on:

- Market fundamentals (population, distance, tourism, passenger demand)
- Competition effects (how many rivals enter)
- Airline-specific size parameters
- Idiosyncratic profit shocks

The model uses a heterogeneous-firm specification where airlines differ in their baseline profitability. Entry probabilities are computed using a sequential entry rule, and parameters are estimated via maximum likelihood (L-BFGS-B).

## Key Results

- **Competition effect (δ):** 1.48 — each additional entrant reduces profit
- **Log-likelihood:** -1908.39
- **Prediction accuracy:** 64.3% for predicting 0, 1, or 2+ entrants per market
- **Passenger demand coefficient:** 0.84 — higher demand strongly encourages entry

See `Berry_Model_Results.pdf` for detailed results and interpretation.

## Installation

### Using Conda (recommended)

```bash
git clone https://github.com/vatsaltailor/berry-airline-entry-model.git
cd berry-airline-entry-model
conda env create -f environment.yml
conda activate berry-entry-model
```

### Using pip

```bash
pip install numpy pandas scipy
```

### Using GitHub Codespaces

Click the **"Open in GitHub Codespaces"** badge above. The environment will be automatically configured.

## Usage

### Basic usage

```bash
python berry_entry_model.py
```

This will:
1. Load and preprocess `Berry_data.csv`
2. Estimate the model via maximum likelihood
3. Print parameter estimates and model fit statistics

### As a Python module

```python
from berry_entry_model import estimate_model

results = estimate_model(data_path='Berry_data.csv', verbose=True)

print(f"Competition effect: {results['delta']:.3f}")
print(f"Log-likelihood: {results['log_likelihood']:.2f}")
print(f"Accuracy: {100 * results['accuracy']:.1f}%")
```

The `results` dictionary contains:
- `beta`: Parameter estimates for market covariates
- `delta`: Competition effect parameter
- `log_likelihood`: Maximized log-likelihood
- `accuracy`: Prediction accuracy (0, 1, or 2+ entrants)
- `n_obs`: Number of markets
- `success`: Whether optimization converged

## Repository Structure

```
.
├── berry_entry_model.py          # Main implementation
├── Berry_data.csv                # Market and airline data (2,742 markets)
├── Berry_Model_Results.pdf       # Writeup with results and interpretation
├── Berry_Model_Results.tex       # LaTeX source for the writeup
├── README.md                     # This file
├── environment.yml               # Conda environment specification
├── .gitignore                    # Git ignore rules
├── .devcontainer/                # GitHub Codespaces configuration
└── airline-entry-model/          # Reference R code and data
    ├── code/                     # R scripts for entry model
    └── data/                     # Raw airline data
```

## Model Specification

The profit function for airline *i* in market *m* is:

```
π_im = V_im - δ N_-im + ε_im
```

where:
- **V_im = X_m' β + s_i**: Deterministic profit index
  - **X_m**: Market covariates (intercept, avg log population, tourism, log distance, log passengers)
  - **s_i**: Airline-specific size parameters [0.1, 0.15, 0.1, 0.15, 0.05, 0.05]
- **δ > 0**: Competition effect (how much each rival reduces profit)
- **N_-im**: Number of rival entrants
- **ε_im ~ N(0,1)**: Idiosyncratic profit shock

Entry probabilities for observing 0, 1, or 2+ entrants are computed using a sequential entry rule where the highest-profit firm moves first.

## Data

`Berry_data.csv` contains 2,742 U.S. airline markets with:

- **Market characteristics:** population at endpoints, distance, tourism, passengers
- **Entry decisions:** Binary indicators for each of 6 airlines (AA, DL, UA, AL, LCC, WN)
- **Outcomes:** Number of entrants per market (0, 1, or 2+)

Missing values in the `passengers` column are imputed with the median.

## Reference

Berry, S. T. (1992). "Estimation of a Model of Entry in the Airline Industry." *Econometrica*, 60(4), 889–917.

## Acknowledgments

This implementation was originally developed as part of coursework at the Gies College of Business, University of Illinois, and has been adapted for public release. I am grateful to [Professor Yixin (Iris) Wang](https://giesbusiness.illinois.edu/profile/yixin-iris-wang) for providing the reference R code (included in `airline-entry-model/`) and the `Berry_data.csv` dataset that made this project possible.

## License

This code is provided for educational and research use.

## Author

Vatsal Mitesh Tailor
