import tensorflow as tf
#import tensorflow_probability as tfp
from data_processing import x_train,x_test,y_train,y_test
import pandas as pd
import matplotlib.pyplot as plt

model = tf.keras.models.Sequential([
     tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(60,activation='relu'),
        tf.keras.layers.Dense(30,activation='relu'),
        tf.keras.layers.Dense(20,activation='relu'),
        tf.keras.layers.Dense(1)
])

# For a probabilistic output too:
'''model_bayes = tf.keras.Sequential([
    tfp.layers.DenseFlipout(64, activation='relu'),
    tfp.layers.DenseFlipout(2),  # outputs params for a distribution
    tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=tf.math.softplus(t[..., 1:]))
    )
])'''


model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])

history = model.fit(x_train,y_train,validation_split=0.2,epochs=50,verbose=0)

performance_df = pd.DataFrame(history.history)

#plt.show()

