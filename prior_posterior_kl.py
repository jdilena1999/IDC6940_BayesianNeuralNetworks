from BNN import model_bayes
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
final = {
    "Layer":[],
    "Kernel Posterior":[],
    "Kernel Prior":[],
    "Bias Posterior":[],
    "Bias Prior":[],
    "KL":[]
}
for layer in model_bayes.layers:
    final["Layer"].append(layer.name)
    final["Kernel Posterior"].append(layer.kernel_posterior)
    final["Kernel Prior"].append(layer.kernel_prior)
    final["Bias Posterior"].append(layer.bias_posterior)
    final["Bias Prior"].append(layer.bias_prior)
    final["KL"].append(float(tf.reduce_sum(
        tfd.kl_divergence(layer.kernel_posterior, layer.kernel_prior)
    )))
model_priors_posteriors_kl = pd.DataFrame(final)
print(model_priors_posteriors_kl)