# Test Data Description

The files in this package are provided solely to demonstrate the structure of model inputs and to verify execution of the released code. They are not the complete observational data used in the manuscript.

## AE-LSTM test data

- `test_data/ae_lstm/autoencoder_grid_test.npy`
  - Synthetic normalized gridded inputs
  - Shape: `(4, 41, 61)`
- `test_data/ae_lstm/lstm_input_test.npy`
  - Synthetic normalized sequence inputs
  - Shape: `(16, 14, 7)`
- `test_data/ae_lstm/lstm_target_test.npy`
  - Synthetic normalized streamflow targets
  - Shape: `(16, 1)`

## Interval-fusion test data

Each CSV contains a time index and a normalized value.

- `observed_streamflow.csv`
- `conformal_lower.csv`
- `conformal_upper.csv`
- `bayesian_lower.csv`
- `bayesian_upper.csv`

These synthetic intervals are constructed only to exercise the interval-fusion and optimization routines.

## Reservoir-regulation test data

- `storage_level_curve.csv`
  - Illustrative storage-water-level relationship
- `level_release_curve.csv`
  - Illustrative water-level-release relationship
- `inflow_test.csv`
  - Illustrative inflow sequence

The reservoir curves are not curves for any reservoir used in the manuscript.

## Restricted research observations

The original precipitation and streamflow observations used in the manuscript were provided by the Water Resources Department of Shandong Province, China, under data-use restrictions. They cannot be publicly redistributed by the authors. Access to those observations is subject to authorization by the original data provider.
