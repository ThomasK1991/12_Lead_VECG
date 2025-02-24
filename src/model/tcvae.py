
import tensorflow as tf
import keras
from model.vae import VAE
class TCVAE(VAE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = tf.Variable(self._coefficients['alpha'], name="alpha", trainable=False)
        self._beta = tf.Variable(self._coefficients['beta'], name="beta", trainable=False)
        self._gamma = tf.Variable(self._coefficients['gamma'], name="gamma", trainable=False)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha.assign(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta.assign(value)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma.assign(value)

    def _loss(self, reconstruction, x, mu, log_var, z):
        print("\n🔍 Debugging `_loss()` function...")

        # ✅ Handle case where `x` is a tuple
        if isinstance(x, tuple):
            print(f"🔹 x is a tuple of length {len(x)}")
            for i, item in enumerate(x):
                print(f"  - Tuple Element {i}: Type = {type(item)}, Shape = {getattr(item, 'shape', 'N/A')}")
            x = x[0]  # Extract tensor from tuple

        # ✅ Ensure `x` is a tensor
        if isinstance(x, tf.Tensor):
            print(f"🔹 Corrected X Shape: {x.shape}, Dtype: {x.dtype}")
        else:
            print(f"⚠️ Warning: X is not a Tensor. Instead, got type: {type(x)}")

        # Compute Loss Components
        mae_loss = tf.keras.losses.MeanAbsoluteError()
        recon_loss = tf.reduce_sum(mae_loss(x, reconstruction))  # Compute MAE loss

        # KL Loss Calculations
        log_qz_x = tf.reduce_sum(self._dist(mu, tf.exp(log_var)).log_prob(z), axis=-1)
        log_prior = tf.reduce_sum(self._dist(tf.zeros_like(z), tf.ones_like(z)).log_prob(z), axis=-1)
        log_qz_prob = self._dist(
            tf.expand_dims(mu, 0), tf.expand_dims(tf.exp(log_var), 0)
        ).log_prob(tf.expand_dims(z, 1))

        log_qz = tf.reduce_logsumexp(self.log_importance_weight_matrix(tf.shape(x)[0]) + tf.reduce_sum(log_qz_prob, axis=2), axis=1)
        log_qz_product = tf.reduce_sum(tf.reduce_logsumexp(tf.expand_dims(self.log_importance_weight_matrix(tf.shape(x)[0]), 2) + log_qz_prob, axis=1), axis=1)

        # ✅ Compute Mutual Info Loss, TC Loss, and Dimension-Wise KL
        mutual_info_loss = tf.reduce_mean(tf.subtract(log_qz_x, log_qz))
        tc_loss = tf.reduce_mean(tf.subtract(log_qz, log_qz_product))
        dimension_wise_kl = tf.reduce_mean(tf.subtract(log_qz_product, log_prior))

        # Final KL loss
        kl_loss = tf.multiply(self._alpha, mutual_info_loss) + tf.multiply(self._beta, tc_loss) + tf.multiply(self._gamma, dimension_wise_kl)

        total_loss = tf.add(recon_loss, kl_loss)

        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl_loss": kl_loss,
            "mi": mutual_info_loss,
            "tc": tc_loss,
            "dw_kl": dimension_wise_kl,
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
        }
