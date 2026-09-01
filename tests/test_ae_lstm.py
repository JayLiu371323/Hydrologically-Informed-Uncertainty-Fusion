from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ae_lstm_model import (
    build_convolutional_autoencoder,
    build_latent_encoder,
    build_ae_lstm,
)

data_dir = ROOT / "test_data" / "ae_lstm"

grid = np.load(data_dir / "autoencoder_grid_test.npy")
x = np.load(data_dir / "lstm_input_test.npy")
y = np.load(data_dir / "lstm_target_test.npy")

autoencoder = build_convolutional_autoencoder(input_shape=(41, 61), latent_dim=3)
latent_encoder = build_latent_encoder(autoencoder)
lstm = build_ae_lstm(window_size=14, features=7)

reconstruction = autoencoder.predict(grid, verbose=0)
latent = latent_encoder.predict(grid, verbose=0)
prediction = lstm.predict(x, verbose=0)

print("Autoencoder input:", grid.shape)
print("Autoencoder reconstruction:", reconstruction.shape)
print("Latent representation:", latent.shape)
print("LSTM input:", x.shape)
print("LSTM target:", y.shape)
print("LSTM prediction:", prediction.shape)

assert reconstruction.shape == grid.shape
assert latent.shape == (grid.shape[0], 3)
assert prediction.shape == y.shape

print("AE-LSTM structural test passed.")
