import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.layers import Dense,Flatten,Input
#from tensorflow.keras.models import Sequential



data = pd.read_csv("radioml_features.csv")
y_col = ['SNR']
x_cols = [x for x in data.columns if x not in y_col]
y = data[y_col]
X = data[x_cols]


# X Features scaled down to mean = 0 and standard_deviation = 1
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(X_transformed,y,test_size=0.3,stratify=y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_transformed.shape[1],)),
    tf.keras.layers.Dense(60,activation='relu'),
    tf.keras.layers.Dense(30,activation='relu'),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)
model_performance = pd.DataFrame(history.history)
model_performance.index.name = 'epoch'
#print(model_performance)
'''model_performance[['mse','val_mse']].plot(title='Training vs Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()'''
#y_pred = model.predict(x_test)
#print(type(y_pred))
#print(type(y_test))
predicted_vs_actual = {
    
}
'''predicted_vs_actual.setdefault("Predicted",y_pred.ravel())
predicted_vs_actual.setdefault("Actual",y_test.to_numpy())
pred_v_actual = pd.DataFrame(predicted_vs_actual)'''

#pred_v_actual[["Predicted","Actual"]].plot(title="Pred vs Actual")
#plt.show()


'''y_values = sorted(list(set(y.values.ravel())))
for y in y_values:
    plt.scatter(x_cols[0],x_cols[1],data=data[data["SNR"].isin([y])],label=y)
plt.legend()
plt.xlabel(x_cols[0])
plt.ylabel(x_cols[1])
plt.show()
#print()'''
'''for c in x_cols:
    plt.scatter(c,"SNR",data=data)
    plt.xlabel(c)
    plt.ylabel("SNR")
    plt.show()'''



