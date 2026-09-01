# Open Research Code and Test Data

This repository contains the author-developed components that can be shared for peer review and reproducibility assessment.

## Included code

- `src/ae_lstm_model.py`
  - Convolutional autoencoder used for input denoising and latent feature extraction
  - LSTM streamflow forecasting model
- `src/interval_fusion.py`
  - Second-order response-function fusion of conformal and Bayesian prediction intervals
  - Three-objective optimization
  - Generation-dependent dynamic objective weighting
  - NSGA-II as the primary optimizer, with optional IBEA, SPEA2, and GDE3 interfaces
- `src/metrics.py`
  - R2, PICP, PINAW, high-flow PICP, and SIS-related metrics
- `src/reservoir_regulation_module.f90`
  - Reservoir-regulation module embedded at a single routing grid cell
  - Storage-water-level and water-level-release relationships
  - Continuity-based storage update and downstream release

## Test data

The `test_data` directory contains synthetic or illustrative test data only. These files are intended to demonstrate input structure and code execution. They are not the complete observational datasets used to generate the results reported in the manuscript.

The original precipitation and streamflow observations used in the study were provided by the Water Resources Department of Shandong Province, China, under data-use restrictions and cannot be redistributed by the authors. Access to the original observations is subject to authorization by the original data provider.

## Quick checks

Python dependencies are listed in `requirements.txt`.

Run:

```bash
python tests/test_ae_lstm.py
python tests/test_interval_fusion.py
```

For the Fortran reservoir module:

```bash
gfortran -ffree-line-length-none src/reservoir_regulation_module.f90 tests/test_reservoir_module.f90 -o test_reservoir
./test_reservoir
```

All tests are designed to use only the included test data.

## Important note

The test data are for code verification only and must not be interpreted as the research observations used in the manuscript.
