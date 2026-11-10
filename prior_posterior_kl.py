from BNN import model_bayes
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
def summarize(dist):
    """Return (mean, std) as floats, averaged over all parameters of the distribution."""
    if dist is None:
        return np.nan, np.nan
    m = tf.reduce_mean(dist.mean()).numpy()
    s = tf.reduce_mean(dist.stddev()).numpy()
    return float(m), float(s)

rows = []
for layer in model_bayes.layers:
    # Skip non-Bayesian layers (Flatten, Dropout, etc.)
    if not hasattr(layer, "kernel_posterior"):
        continue

    k_post_mean, k_post_std = summarize(layer.kernel_posterior)
    k_prior_mean, k_prior_std = summarize(layer.kernel_prior)
    b_post_mean, b_post_std = summarize(getattr(layer, "bias_posterior", None))
    b_prior_mean, b_prior_std = summarize(getattr(layer, "bias_prior", None))

    kl = float(tf.reduce_sum(
        tfd.kl_divergence(layer.kernel_posterior, layer.kernel_prior)
    ))

    rows.append({
        "Layer": layer.name,
        "Kernel Post Mean": k_post_mean,
        "Kernel Post Std": k_post_std,
        "Kernel Prior Mean": k_prior_mean,
        "Kernel Prior Std": k_prior_std,
        #"Bias Post Mean": b_post_mean,
        #"Bias Post Std": b_post_std,
        #"Bias Prior Mean": b_prior_mean,
        #"Bias Prior Std": b_prior_std,
        "KL": kl,
    })

model_priors_posteriors_kl = pd.DataFrame(rows)
#print(model_priors_posteriors_kl)