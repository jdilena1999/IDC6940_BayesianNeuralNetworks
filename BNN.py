# How to start a python 3.11 virtual environment:
#   deactivate
#   rm -rf .venv
#   /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv .venv
#   source .venv/bin/activate



# Use negative log-likelihood (NLL) as the training loss, 
# since your model predicts a probability distribution. 
# You can still calculate MSE and MAE separately using the distribution's predicted mean.
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_probability as tfp
from data_processing import x_train,x_test,y_train,y_test
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

'''model = tf.keras.models.Sequential([
     tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(60,activation='relu'),
        tf.keras.layers.Dense(30,activation='relu'),
        tf.keras.layers.Dense(20,activation='relu'),
        tf.keras.layers.Dense(1)
])'''

# For a probabilistic output too:
model_bayes = tf.keras.Sequential([
    tfp.layers.DenseFlipout(128, activation='relu'),
    tfp.layers.DenseFlipout(64, activation='relu'),
    tfp.layers.DenseFlipout(50, activation='relu'),
    tfp.layers.DenseFlipout(50, activation='relu'),
    tfp.layers.DenseFlipout(32, activation='relu'),

    tfp.layers.DenseFlipout(2),  # outputs params for a distribution
    #tfp.layers.DistributionLambda(
        #lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:]))
    #)
])


# Train using negative log-likelihood
tfd = tfp.distributions

# Match y_true's shape to the single predicted target per example.
def unpack_predictions(y_pred):
    loc = y_pred[..., :1]
    scale = 1e-3 + tf.nn.softplus(y_pred[..., 1:2])
    return loc, scale

def gaussian_nll(y_true, y_pred):
    loc, scale = unpack_predictions(y_pred)
    y_true = tf.reshape(tf.cast(y_true, loc.dtype), tf.shape(loc))

    dist = tfd.Normal(loc=loc, scale=scale)
    return -tf.reduce_mean(dist.log_prob(y_true))

def mean_mse(y_true, y_pred):
    loc, _ = unpack_predictions(y_pred)
    y_true = tf.reshape(tf.cast(y_true, loc.dtype), tf.shape(loc))
    return tf.reduce_mean(tf.square(y_true - loc))

def mean_mae(y_true, y_pred):
    loc, _ = unpack_predictions(y_pred)
    y_true = tf.reshape(tf.cast(y_true, loc.dtype), tf.shape(loc))
    return tf.reduce_mean(tf.abs(y_true - loc))

model_bayes.compile(
    optimizer=tf.keras.optimizers.Adam(
    learning_rate=1e-4
),
    loss=gaussian_nll,
    metrics=[mean_mae, mean_mse]
)
history = model_bayes.fit(x_train,y_train,validation_split=0.2,epochs=200,verbose=0,shuffle=True)

performance_df = pd.DataFrame(history.history)

y_pred = model_bayes.predict(x_test,verbose=0)
y_pred_means,y_pred_std = unpack_predictions(y_pred)

actual_v_predicted = {
    
}
#
actual_v_predicted.setdefault("Actual",y_test.to_numpy())
actual_v_predicted.setdefault("Predicted Mean",np.array(y_pred_means).ravel())
actual_v_predicted.setdefault("Predicted Std",np.array(y_pred_std).ravel())


a_v_p = pd.DataFrame(actual_v_predicted)
#print(a_v_p)
#print(actual_v_predicted)
#print(a_v_p)
#print(performance_df)
fig,ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=performance_df,x=performance_df.index.values,y='mean_mse',color='red',label="Training MSE",ax=ax)
sns.lineplot(data=performance_df,x=performance_df.index.values,y='val_mean_mse',color='blue',label="Validation MSE",ax=ax)

plt.title(f"Training vs Validation MSE")
plt.savefig(f"./images/Training vs Validation MSE.png")
plt.show()


fig,ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=performance_df,x=performance_df.index.values,y='mean_mae',color='red',label="Training MAE",ax=ax)
sns.lineplot(data=performance_df,x=performance_df.index.values,y='val_mean_mae',color='blue',label="Validation MAE",ax=ax)

plt.title(f"Training vs Validation MAE")
plt.savefig(f"./images/Training vs Validation MAE.png")
plt.show()



fig,ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=performance_df,x=performance_df.index.values,y='loss',color='red',label="Training Loss",ax=ax)
sns.lineplot(data=performance_df,x=performance_df.index.values,y='val_loss',color='blue',label="Validation Loss",ax=ax)

plt.title(f"Training vs Validation Loss")
plt.savefig(f"./images/Training vs Validation Loss (Negative Log-Loss).png")
plt.show()



# Overall Means and Standard Devs:

T = 200
means,stdevs = [],[]
for _ in range(T):
    y_pred = model_bayes.predict(x_test,verbose=0)
    mean,std = unpack_predictions(y_pred)
    means.append(mean.ravel())
    stdevs.append(std.numpy().ravel())
means = np.stack(means)
stdevs = np.stack(stdevs)

final_mean_pred = means.mean(axis=0)
total_sd = np.sqrt(means.var(axis=0) + (stdevs**2).mean(axis=0))  # epistemic + aleatoric
'''print(final_mean_pred.shape())
print(total_sd.shape())'''
#print(final_mean_pred)
#print(total_sd)

# All Distributions:
rand_xs = np.linspace(final_mean_pred-4*total_sd,final_mean_pred+4*total_sd,1000)
rand_ys = np.array(stats.Normal(mu=final_mean_pred,sigma=total_sd).pdf(rand_xs))

plt.plot(rand_xs,rand_ys)
plt.title("All Distributions")
plt.savefig(f"./images/All_Distributions.png")
plt.show()

# Avg Distribution:

rand_xs = np.linspace(final_mean_pred.mean()-4*total_sd.mean(),final_mean_pred.mean()+4*total_sd.mean(),1000)
rand_ys = np.array(stats.Normal(mu=final_mean_pred.mean(),sigma=total_sd.mean()).pdf(rand_xs))

plt.plot(rand_xs,rand_ys)
plt.title(f"Average Distibution\nMean: {final_mean_pred.mean():,.2f} | St. Dev: {total_sd.mean():,.2f}")
for sd,c,label in zip([x*total_sd.mean() + final_mean_pred.mean() for x in [1,2,3]],["green",'yellow','red'],["1 SD","2 SD","3 SD"]):
    plt.axvline(sd,0,1,linestyle='--',color=c)
    plt.axvline(-sd,0,1,linestyle='--',color=c)
plt.legend()
plt.savefig(f"./images/Average_Distribution.png")
plt.show()


