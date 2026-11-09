from BNN import model_bayes
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
final = {
    "Layer":[],
    "Kernal Posterior":[],
    "Kernal Prior":[],
    "Bias Posterior":[],
    "Bias Prior":[],
    "KL":[]
}
for layer in model_bayes.layers:
    final["Layer"].append(layer.name)
    final["Kernal Posterior"].append(layer.kernal_posterior)
    final["Kernal Prior"].append(layer.kernal_prior)
    final["Bias Posterior"].append(layer.bias_posterior)
    final["Bias Prior"].append(layer.bias_prior)
    final["KL"].append(float(tf.reduce_sum(layer.losses)))
model_priors_posteriors_kl = pd.DataFrame(final)
print(final)