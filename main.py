#Предобработка данных
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('heart.csv')
df.head()
df.dtypes

df.loc[df.duplicated(subset=["age","sex","cp","trestbps","chol","fbs","exang"])]

from sklearn.model_selection import train_test_split
X = df.copy().drop(['target'], axis=1)
y = df.copy()['target']

x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Модель
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(50, input_shape=(x_train.shape[1],), activation= "relu"))
model.add(Dense(75, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer='Adam', loss = "binary_crossentropy",metrics= ["accuracy"])
model.fit(x=x_train, y=y_train, epochs=1500, validation_data=(x_test, y_test),verbose=1)

model.save('Heart.h5')

