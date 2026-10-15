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
    tfp.layers.DenseFlipout(32, activation='relu'),

    tfp.layers.DenseFlipout(2),  # outputs params for a distribution
    tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:]))
    )
])


# Train using negative log-likelihood
def nll(y_true, y_pred):
    return -y_pred.log_prob(tf.cast(y_true, y_pred.dtype))

# Evaluate the predicted mean with standard regression metrics
def mean_mse(y_true, y_pred):
    mean = y_pred.mean()
    return tf.reduce_mean(tf.square(
        tf.cast(y_true, mean.dtype) - mean
    ))

def mean_mae(y_true, y_pred):
    mean = y_pred.mean()
    return tf.reduce_mean(tf.abs(
        tf.cast(y_true, mean.dtype) - mean
    ))

model_bayes.compile(
    optimizer='adam',
    loss=nll,
    metrics=[mean_mae, mean_mse]
)
history = model_bayes.fit(x_train,y_train,validation_split=0.2,epochs=50,verbose=0)

performance_df = pd.DataFrame(history.history)

y_pred = model_bayes.predict(x_test,verbose=0)

actual_v_predicted = {
    
}
#
actual_v_predicted.setdefault("Actual",y_test.to_numpy())
actual_v_predicted.setdefault("Predicted",y_pred.ravel())

a_v_p = pd.DataFrame(actual_v_predicted)
#print(actual_v_predicted)
#print(a_v_p)
#print(performance_df)
fig,ax = plt.subplots()
sns.lineplot(data=performance_df,x=performance_df.index.values,y='mse',color='red')
sns.lineplot(data=performance_df,x=performance_df.index.values,y='val_mse',color='blue')
plt.show()
#plt.show()

