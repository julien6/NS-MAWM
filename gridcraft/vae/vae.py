import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_config import (
  BLOCK_CLASSES,
  ENTITY_CLASSES,
  GRID_CELLS,
  GRID_FEATURES,
  OBS_SIZE,
  SELF_FEATURES,
  TERRAIN_CLASSES,
  Z_SIZE,
)


class GridcraftVAE:
  def __init__(self, z_size=Z_SIZE, hidden_size=512, learning_rate=1e-3, kl_tolerance=0.5):
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

  def decode_tabular(self, z):
    decoded = self.decode(np.asarray(z, dtype=np.float32).reshape(-1, self.z_size))
    result = []
    for row in decoded:
      result.append(vector_to_tabular(row))
    return result[0] if len(result) == 1 else result

  def train_batch(self, x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
      mu, logvar = self.encoder(x, training=True)
      eps = tf.random.normal(tf.shape(mu))
      z = mu + tf.exp(logvar / 2.0) * eps
      y = self.decoder(z, training=True)
      terrain_loss = categorical_plane_loss(x, y, 0, TERRAIN_CLASSES)
      block_loss = categorical_plane_loss(x, y, GRID_CELLS * TERRAIN_CLASSES, BLOCK_CLASSES)
      entity_loss = categorical_plane_loss(x, y, GRID_CELLS * (TERRAIN_CLASSES + BLOCK_CLASSES), ENTITY_CLASSES)
      grid_loss = terrain_loss + block_loss + entity_loss
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


def categorical_plane_loss(labels, logits, offset, depth):
  label_plane = tf.reshape(labels[:, offset:offset + GRID_CELLS * depth], (-1, GRID_CELLS, depth))
  logit_plane = tf.reshape(logits[:, offset:offset + GRID_CELLS * depth], (-1, GRID_CELLS, depth))
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_plane, logits=logit_plane))


def vector_to_tabular(vector):
  vector = np.asarray(vector, dtype=np.float32)
  cursor = 0
  terrain = vector[cursor:cursor + GRID_CELLS * TERRAIN_CLASSES].reshape(GRID_CELLS, TERRAIN_CLASSES).argmax(axis=1)
  cursor += GRID_CELLS * TERRAIN_CLASSES
  blocks = vector[cursor:cursor + GRID_CELLS * BLOCK_CLASSES].reshape(GRID_CELLS, BLOCK_CLASSES).argmax(axis=1)
  cursor += GRID_CELLS * BLOCK_CLASSES
  entities = vector[cursor:cursor + GRID_CELLS * ENTITY_CLASSES].reshape(GRID_CELLS, ENTITY_CLASSES).argmax(axis=1)
  self_vec = vector[GRID_FEATURES:GRID_FEATURES + SELF_FEATURES]
  hp_hunger = np.clip(np.rint(self_vec[:2] * 20.0), 0, 20)
  inventory = np.clip(np.rint(self_vec[2:] * 10.0), 0, 99)
  return {
    "grid": np.stack([
      terrain.reshape(int(np.sqrt(GRID_CELLS)), int(np.sqrt(GRID_CELLS))),
      blocks.reshape(int(np.sqrt(GRID_CELLS)), int(np.sqrt(GRID_CELLS))),
      entities.reshape(int(np.sqrt(GRID_CELLS)), int(np.sqrt(GRID_CELLS))),
    ]).astype(np.int8),
    "self": np.concatenate([hp_hunger, inventory]).astype(np.int16),
  }
