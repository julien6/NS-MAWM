import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def reset_graph():
  tf.keras.backend.clear_session()


class ConvVAE(object):
  def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
    self.z_size = z_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.is_training = is_training
    self.kl_tolerance = kl_tolerance
    self.reuse = reuse
    self._build_model()

  def _build_model(self):
    inputs = tf.keras.Input(shape=(64, 64, 3), name="x")
    h = tf.keras.layers.Conv2D(32, 4, strides=2, activation="relu", padding="valid", name="enc_conv1")(inputs)
    h = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu", padding="valid", name="enc_conv2")(h)
    h = tf.keras.layers.Conv2D(128, 4, strides=2, activation="relu", padding="valid", name="enc_conv3")(h)
    h = tf.keras.layers.Conv2D(256, 4, strides=2, activation="relu", padding="valid", name="enc_conv4")(h)
    h = tf.keras.layers.Reshape((2 * 2 * 256,), name="enc_flatten")(h)
    mu = tf.keras.layers.Dense(self.z_size, name="enc_fc_mu")(h)
    logvar = tf.keras.layers.Dense(self.z_size, name="enc_fc_log_var")(h)
    self.encoder = tf.keras.Model(inputs, [mu, logvar], name="conv_vae_encoder")

    z_inputs = tf.keras.Input(shape=(self.z_size,), name="z")
    h = tf.keras.layers.Dense(4 * 256, name="dec_fc")(z_inputs)
    h = tf.keras.layers.Reshape((1, 1, 4 * 256), name="dec_reshape")(h)
    h = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, activation="relu", padding="valid", name="dec_deconv1")(h)
    h = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, activation="relu", padding="valid", name="dec_deconv2")(h)
    h = tf.keras.layers.Conv2DTranspose(32, 6, strides=2, activation="relu", padding="valid", name="dec_deconv3")(h)
    y = tf.keras.layers.Conv2DTranspose(3, 6, strides=2, activation="sigmoid", padding="valid", name="dec_deconv4")(h)
    self.decoder = tf.keras.Model(z_inputs, y, name="conv_vae_decoder")
    self.model = tf.keras.Model(inputs, self.decoder(mu), name="conv_vae")

  @property
  def _ordered_layers(self):
    return [
      self.encoder.get_layer("enc_conv1"),
      self.encoder.get_layer("enc_conv2"),
      self.encoder.get_layer("enc_conv3"),
      self.encoder.get_layer("enc_conv4"),
      self.encoder.get_layer("enc_fc_mu"),
      self.encoder.get_layer("enc_fc_log_var"),
      self.decoder.get_layer("dec_fc"),
      self.decoder.get_layer("dec_deconv1"),
      self.decoder.get_layer("dec_deconv2"),
      self.decoder.get_layer("dec_deconv3"),
      self.decoder.get_layer("dec_deconv4"),
    ]

  def close_sess(self):
    pass

  def encode(self, x):
    mu, logvar = self.encode_mu_logvar(x)
    return mu + np.exp(logvar / 2.0) * np.random.randn(*logvar.shape)

  def encode_mu_logvar(self, x):
    mu, logvar = self.encoder(np.asarray(x, dtype=np.float32), training=False)
    return mu.numpy(), logvar.numpy()

  def decode(self, z):
    return self.decoder(np.asarray(z, dtype=np.float32), training=False).numpy()

  def get_model_params(self):
    model_params = []
    model_shapes = []
    model_names = []
    for layer in self._ordered_layers:
      weights = layer.get_weights()
      if len(weights) != 2:
        continue
      kernel, bias = weights
      for name, value in [(layer.name + "/kernel:0", kernel), (layer.name + "/bias:0", bias)]:
        model_names.append(name)
        model_params.append(np.round(value * 10000).astype(int).tolist())
        model_shapes.append(value.shape)
    return model_params, model_shapes, model_names

  def get_random_model_params(self, stdev=0.5):
    _, mshape, _ = self.get_model_params()
    return [np.random.standard_cauchy(s) * stdev for s in mshape]

  def set_model_params(self, params):
    idx = 0
    for layer in self._ordered_layers:
      weights = layer.get_weights()
      if len(weights) != 2:
        continue
      kernel_shape = weights[0].shape
      bias_shape = weights[1].shape
      kernel = np.asarray(params[idx], dtype=np.float32) / 10000.0
      bias = np.asarray(params[idx + 1], dtype=np.float32) / 10000.0
      assert kernel.shape == kernel_shape, "inconsistent kernel shape"
      assert bias.shape == bias_shape, "inconsistent bias shape"
      layer.set_weights([kernel, bias])
      idx += 2

  def load_json(self, jsonfile='vae.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)

  def save_json(self, jsonfile='vae.json'):
    model_params, _, _ = self.get_model_params()
    with open(jsonfile, 'wt') as outfile:
      json.dump(model_params, outfile, sort_keys=True, indent=0, separators=(',', ': '))

  def set_random_params(self, stdev=0.5):
    self.set_model_params(self.get_random_model_params(stdev))

  def save_model(self, model_save_path):
    os.makedirs(model_save_path, exist_ok=True)
    self.model.save(os.path.join(model_save_path, "vae.keras"))

  def load_checkpoint(self, checkpoint_path):
    self.model = tf.keras.models.load_model(checkpoint_path)
