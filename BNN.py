import tensorflow as tf
import tensorflow_probability as tfp

model = tf.keras.Sequential([
    tfp.layers.DenseFlipout(64, activation='relu'),
    tfp.layers.DenseFlipout(64, activation='relu'),
    tfp.layers.DenseFlipout(1),
])

# For a probabilistic output too:
model_prob_output = tf.keras.Sequential([
    tfp.layers.DenseFlipout(64, activation='relu'),
    tfp.layers.DenseFlipout(2),  # outputs params for a distribution
    tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:]))
    )
])