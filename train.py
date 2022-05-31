"""Script to train RNN models to predict polynomial additions."""

import numpy as np
import pandas as pd
import sys
import tensorflow as tf


BASE_DIR = "data"


def load_strat(dist, strat):
    strat = strat + "_123" if strat == "random" else strat
    return pd.read_csv(BASE_DIR + f"/stats/{dist}/{dist}_{strat}.csv")


def load_array(dist):
    return np.load(BASE_DIR + f"/stats/{dist}/{dist}.npy", allow_pickle=True)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: train.py <dist> <units>")
        sys.exit()
    dist = sys.argv[1]
    units = int(sys.argv[2])

    X = load_array(dist)
    y = load_strat(dist, "degree").PolynomialAdditions.to_numpy()
    train_size = int(0.8 * len(X))
    valid_size = int(0.1 * len(X))
    X_train = X[:train_size]
    X_valid = X[train_size:train_size+valid_size]
    y_train = y[:train_size]
    y_valid = y[train_size:train_size+valid_size]

    input_size = 16 if dist.startswith('toric') else 6
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, input_size)),
        tf.keras.layers.Masking(mask_value=-1),
        tf.keras.layers.GRU(units),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss='mse', optimizer='adam')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=BASE_DIR+f"/models/{dist}/{units}",
                                           save_best_only=True,
                                           monitor="val_loss"),
        tf.keras.callbacks.TensorBoard(log_dir=BASE_DIR+f"/logs/{dist}/{units}"),
    ]
    
    if dist.startswith('toric'):
        train_ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_generator(lambda: X_train, tf.int32),
            tf.data.Dataset.from_tensor_slices(y_train.astype(np.float32)),
        ))
        padded_shapes = ([None, X[0].shape[1]], [])
        padding_values = (tf.constant(-1, dtype=tf.int32),
                          tf.constant(0.0, dtype=tf.float32))
        train_ds = train_ds.padded_batch(32,
                                         padded_shapes=padded_shapes,
                                         padding_values=padding_values)

        valid_ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_generator(lambda: X_valid, tf.int32),
            tf.data.Dataset.from_tensor_slices(y_valid.astype(np.float32)),
        ))
        padded_shapes = ([None, X[0].shape[1]], [])
        padding_values = (tf.constant(-1, dtype=tf.int32),
                          tf.constant(0.0, dtype=tf.float32))
        valid_ds = valid_ds.padded_batch(32,
                                         padded_shapes=padded_shapes,
                                         padding_values=padding_values)
        
        history = model.fit(train_ds, epochs=1000,
                            validation_data=valid_ds,
                            callbacks=callbacks)
    else:
        history = model.fit(X_train, y_train, epochs=1000,
                            validation_data=(X_valid, y_valid),
                            callbacks=callbacks)
