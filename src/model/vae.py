import tensorflow as tf
import tensorflow_probability as tfp
import keras
import numpy as np
import math

from utils.helper import Helper

tfd = tfp.distributions


class VAE(tf.keras.Model):

    def __init__(self, encoder, decoder, coefficients, dataset_size, dist=tfd.Normal, **kwargs):
        super().__init__(**kwargs)

        self._encoder = encoder
        self._decoder = decoder
        self._coefficients = coefficients
        self._dataset_size = dataset_size
        self._dist = dist
        self._mi = tf.Variable(0.0, name="mi", trainable=False)
        self._mi_val = tf.Variable(0.0, name="mi_val", trainable=False)
        self._loss_tracker = keras.metrics.Mean(name="loss")
        self._recon_tracker = keras.metrics.Mean(name="recon")
        self._kl_tracker = keras.metrics.Mean(name="kl")
        self._mi_tracker = keras.metrics.Mean(name="mi")
        self._tc_tracker = keras.metrics.Mean(name="tc")
        self._dwkl_tracker = keras.metrics.Mean(name="dw_kl")

    @property
    def metrics(self):
        return [
            self._loss_tracker,
            self._recon_tracker,
            self._kl_tracker,
            self._mi_tracker,
            self._tc_tracker,
            self._dwkl_tracker
        ]

    @staticmethod
    def reparameterize(mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return tf.add(mean, tf.multiply(eps, tf.exp(log_var * 0.5)), name="sampled_latent_variable")

    def get_config(self):
        """ Serialize VAE configuration for saving """
        config = super().get_config()
        config.update({
            "encoder": keras.saving.serialize_keras_object(self._encoder),
            "decoder": keras.saving.serialize_keras_object(self._decoder),
            "coefficients": self._coefficients,
            "dataset_size": self._dataset_size,
            "dist": self._dist.__name__ if hasattr(self._dist, "__name__") else str(self._dist),  # ✅ Ensure it's serializable
        })
        return config
    @classmethod
    def from_config(cls, config):
        """ Deserialize VAE from config """
        
        # ✅ Restore encoder & decoder
        encoder = keras.saving.deserialize_keras_object(config["encoder"])
        decoder = keras.saving.deserialize_keras_object(config["decoder"])
        
        # ✅ Restore coefficients
        coefficients = config["coefficients"]

        # ✅ Restore dataset size
        dataset_size = config["dataset_size"]

        # ✅ Convert `_dist` string back to a distribution class
        dist_str = config["dist"]
        if dist_str == "Normal":
            dist = tfd.Normal
        else:
            raise ValueError(f"❌ Unknown distribution type in config: {dist_str}")

        # ✅ Return reconstructed VAE
        return cls(encoder, decoder, coefficients, dataset_size, dist=dist)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def encode(self, inputs):
        z_mean, z_log_var = self._encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z

    @tf.function
    def decode(self, z):
        reconstructed = self._decoder(z)
        return reconstructed

    @property
    def mi(self):
        return self._mi

    @mi.setter
    def mi(self, value):
        self._mi.assign(value)

    @property
    def mi_val(self):
        return self._mi_val

    @mi.setter
    def mi_val(self, value):
        self._mi_val.assign(value)

    def call(self, inputs):
        _, _, z = self.encode(inputs)
        reconstruction = self.decode(z)
        return reconstruction

    @tf.function
    def train_step(self, data):
        print(type(data))
        with (tf.GradientTape() as tape):
            z_mean, z_log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            loss_value = self._loss(reconstruction, data, z_mean, z_log_var, z)
        grads = tape.gradient(loss_value['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._loss_tracker.update_state(loss_value['loss'])
        self._recon_tracker.update_state(loss_value['recon'])
        self._kl_tracker.update_state(loss_value['kl_loss'])
        self._mi_tracker.update_state(loss_value['mi'])
        self._tc_tracker.update_state(loss_value['tc'])
        self._dwkl_tracker.update_state(loss_value['dw_kl'])
        self._mi.assign(loss_value['mi'])
        return loss_value
    
    @tf.function
    def test_step(self, data):
        print("\n🔍 Debugging test_step()")

        # 🔹 Check the type of `data`
        print(f"🔹 Type of `data`: {type(data)}")
        
        if isinstance(data, tuple):
            print("✅ Data is a tuple")
            print(f"🔹 Tuple length: {len(data)}")
            print(f"🔹 First element type: {type(data[0])}")

            if isinstance(data[0], tf.Tensor):
                print(f"🔹 First element shape: {data[0].shape}")
        
        if isinstance(data, tf.Tensor):
            print("❌ Data is a Tensor, but expected a tuple")
            print(f"🔹 Tensor shape: {data.shape}")

        # 🔹 Ensure correct format before encoding
        if isinstance(data, tuple):
            data = data[0]  # Extract the actual tensor from the tuple if needed
            print(f"✅ Extracted Tensor Shape: {data.shape}")

        # 🔹 Step 1: Encoding
        print("\n🔹 Encoding Step")
        z_mean, z_log_var = self._encoder(data)
        print(f"✅ Encoded z_mean shape: {z_mean.shape}, dtype: {z_mean.dtype}")
        print(f"✅ Encoded z_log_var shape: {z_log_var.shape}, dtype: {z_log_var.dtype}")

        # 🔹 Step 2: Reparameterization Trick
        print("\n🔹 Reparameterization Step")
        z = self.reparameterize(z_mean, z_log_var)
        print(f"✅ Latent Variable z shape: {z.shape}, dtype: {z.dtype}")

        # 🔹 Step 3: Decoding
        print("\n🔹 Decoding Step")
        reconstruction = self._decoder(z)
        print(f"✅ Decoded Reconstruction shape: {reconstruction.shape}, dtype: {reconstruction.dtype}")

        # 🔹 Step 4: Loss Calculation
        print("\n🔹 Loss Calculation Step")
        loss_value = self._loss(reconstruction, data, z_mean, z_log_var, z)

        # 🔹 Store Mutual Information Value
        print("\n🔹 Storing MI Value")
        self._mi_val.assign(loss_value['mi'])
        
        print("\n✅ Test Step Completed Successfully\n")
        
        return loss_value


    def log_normal_pdf(self, sample, mean, logvar):
        log2pi = tf.math.log(2. * np.pi)
        return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

    def log_importance_weight_matrix_iso(self, batch_size):
        """
        CRE: TF adapted version of (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """
        N = tf.constant(self._dataset_size)
        M = tf.math.subtract(batch_size, 1)
        strat_weight = tf.divide(tf.math.subtract(N, M), tf.math.multiply(N, M))
        new_column1 = tf.cast(tf.fill((batch_size, 1), tf.divide(1, N)), tf.float32)
        new_column2 = tf.cast(tf.fill((batch_size, 1), strat_weight), tf.float32)

        W = tf.divide(tf.ones([batch_size, batch_size]), tf.cast(M, tf.float32))
        W = tf.concat([new_column1, new_column2, W[:, 2:]], axis=1)
        W = tf.tensor_scatter_nd_update(W, [[M - 1, 0]], [strat_weight])

        return tf.math.log(W)

    def log_importance_weight_matrix(self, batch_size):
        N = tf.cast(tf.constant(self._dataset_size), dtype=tf.float32)
        B = tf.cast(tf.math.subtract(batch_size, 1), dtype=tf.float32)
        W = tf.multiply(tf.ones([batch_size, batch_size]),
                        tf.divide(tf.subtract(N, 1), tf.multiply(N, tf.subtract(B, 1))))
        W = tf.linalg.set_diag(W, tf.multiply(tf.divide(1.0, N), tf.ones([batch_size])))

        return tf.math.log(W)

    def compute_information_gain(self, data):
        # Approximate the mutual information between x and z
        # [x_batch, nz]

        mu, log_var = self._encoder.predict(Helper.data_generator(data, method='stop'))
        z = self.reparameterize(mu, log_var)
        size_batch, nz = mu.shape
        logiw_mat = self.log_importance_weight_matrix(size_batch)

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*pi) - 0.5*(1+logvar).sum(axis=-1)
        neg_entropy = np.mean(np.sum(-0.5 * nz * np.log(2 * math.pi) - 0.5 * (1 + log_var), axis=-1))

        log_qz_prob = self._dist(
            tf.expand_dims(mu, 0), tf.expand_dims(tf.exp(log_var), 0),
        ).log_prob(tf.expand_dims(z, 1))
        log_qz = tf.reduce_logsumexp(logiw_mat + tf.reduce_sum(log_qz_prob, axis=2), axis=1)

        mi = neg_entropy - np.mean(log_qz, axis=-1)

        tf.print('\n\n')
        return mi
