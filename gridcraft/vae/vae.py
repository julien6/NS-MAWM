import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_config import GRID_FEATURES, OBS_SIZE, Z_SIZE


class GridcraftVAE:
  def __init__(self, z_size=Z_SIZE, hidden_size=256, learning_rate=1e-3, kl_tolerance=0.5):
    self.z_size = z_size
    self.hidden_size = hidden_size
    self.kl_tolerance = kl_tolerance
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)
    self._build()

  def _build(self):
    x = tf.keras.Input(shape=(OBS_SIZE,))
    h = tf.keras.layers.Dense(self.hidden_size, activation="relu", name="enc_dense1")(x)
    h = tf.keras.layers.Dense(self.hidden_size, activation="relu", name="enc_dense2")(h)
    mu = tf.keras.layers.Dense(self.z_size, name="enc_mu")(h)
    logvar = tf.keras.layers.Dense(self.z_size, name="enc_logvar")(h)
    self.encoder = tf.keras.Model(x, [mu, logvar], name="gridcraft_encoder")

    z = tf.keras.Input(shape=(self.z_size,))
    h = tf.keras.layers.Dense(self.hidden_size, activation="relu", name="dec_dense1")(z)
    h = tf.keras.layers.Dense(self.hidden_size, activation="relu", name="dec_dense2")(h)
    y = tf.keras.layers.Dense(OBS_SIZE, name="dec_out")(h)
    self.decoder = tf.keras.Model(z, y, name="gridcraft_decoder")

  def encode_mu_logvar(self, x):
    mu, logvar = self.encoder(np.asarray(x, dtype=np.float32), training=False)
    return mu.numpy(), logvar.numpy()

  def encode(self, x):
    mu, logvar = self.encode_mu_logvar(x)
    return mu + np.exp(logvar / 2.0) * np.random.randn(*logvar.shape)

  def decode(self, z):
    return self.decoder(np.asarray(z, dtype=np.float32), training=False).numpy()

  def train_batch(self, x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
      mu, logvar = self.encoder(x, training=True)
      eps = tf.random.normal(tf.shape(mu))
      z = mu + tf.exp(logvar / 2.0) * eps
      y = self.decoder(z, training=True)
      grid_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=x[:, :GRID_FEATURES], logits=y[:, :GRID_FEATURES]))
      self_loss = tf.reduce_mean(tf.square(x[:, GRID_FEATURES:] - y[:, GRID_FEATURES:]))
      kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
      kl = tf.reduce_mean(tf.maximum(kl, self.kl_tolerance * self.z_size))
      loss = grid_loss + self_loss + kl
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    grads = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(grads, variables))
    return float(loss.numpy()), float(grid_loss.numpy()), float(self_loss.numpy()), float(kl.numpy())

  @property
  def _layers(self):
    return [
      self.encoder.get_layer("enc_dense1"),
      self.encoder.get_layer("enc_dense2"),
      self.encoder.get_layer("enc_mu"),
      self.encoder.get_layer("enc_logvar"),
      self.decoder.get_layer("dec_dense1"),
      self.decoder.get_layer("dec_dense2"),
      self.decoder.get_layer("dec_out"),
    ]

  def get_model_params(self):
    params = []
    shapes = []
    names = []
    for layer in self._layers:
      for suffix, value in zip(("kernel", "bias"), layer.get_weights()):
        params.append(np.asarray(value).tolist())
        shapes.append(value.shape)
        names.append(f"{layer.name}/{suffix}")
    return params, shapes, names

  def set_model_params(self, params):
    idx = 0
    for layer in self._layers:
      weights = layer.get_weights()
      new_weights = []
      for weight in weights:
        value = np.asarray(params[idx], dtype=np.float32)
        assert value.shape == weight.shape, f"shape mismatch for {layer.name}"
        new_weights.append(value)
        idx += 1
      layer.set_weights(new_weights)

  def save_json(self, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params, _, _ = self.get_model_params()
    with open(path, "w") as f:
      json.dump(params, f)

  def load_json(self, path):
    with open(path) as f:
      self.set_model_params(json.load(f))


ConvVAE = GridcraftVAE
