"""
Author-developed AE-LSTM components for streamflow forecasting.

Observational data and pretrained weights are not included.
"""

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Conv2D,
    Conv2DTranspose,
    AveragePooling2D,
    Flatten,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_convolutional_autoencoder(input_shape=(41, 61), kernel_size=(3, 3), latent_dim=3):
    """Build the convolutional autoencoder used for gridded-input denoising."""
    inputs = Input(shape=input_shape, name="grid_input")
    x = tf.expand_dims(inputs, axis=-1)

    x = Conv2D(32, kernel_size, padding="same")(x)
    x = Conv2D(16, kernel_size, padding="same")(x)
    x = AveragePooling2D(pool_size=input_shape)(x)

    x = Flatten()(x)
    latent = Dense(latent_dim, name="latent_out", activation="sigmoid")(x)

    x = Dense(16)(latent)
    x = Dense(32)(x)
    x = Dense(64)(x)
    x = Dense(input_shape[0] * input_shape[1])(x)
    x = Reshape((input_shape[0], input_shape[1], 1))(x)

    x = Conv2DTranspose(16, kernel_size, padding="same")(x)
    x = Conv2DTranspose(32, kernel_size, padding="same")(x)
    x = Conv2DTranspose(1, kernel_size, padding="same", activation="relu")(x)
    outputs = tf.squeeze(x, axis=-1)

    model = Model(inputs=inputs, outputs=outputs, name="convolutional_autoencoder")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_latent_encoder(autoencoder):
    """Return a model that extracts the autoencoder latent representation."""
    return Model(
        inputs=autoencoder.inputs,
        outputs=autoencoder.get_layer("latent_out").output,
        name="latent_encoder",
    )


def build_ae_lstm(window_size=14, features=7):
    """Build the LSTM forecasting network used after denoising/feature extraction."""
    inputs = Input(shape=(window_size, features), name="sequence_input")

    x = LSTM(
        64,
        return_sequences=True,
        recurrent_regularizer=regularizers.l2(0.1),
    )(inputs)
    x = LSTM(
        64,
        return_sequences=True,
        recurrent_regularizer=regularizers.l2(0.1),
    )(x)
    x = LSTM(
        64,
        return_sequences=False,
        recurrent_regularizer=regularizers.l2(0.1),
    )(x)

    x = Dense(32)(x)
    outputs = Dense(1, name="streamflow_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ae_lstm")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
