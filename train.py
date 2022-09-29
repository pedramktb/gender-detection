import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

def plot(history, variable, variable1):
    plt.plot(range(len(history[variable])), history[variable])
    plt.plot(range(len(history[variable1])), history[variable1])
    plt.title(variable)
    plt.legend([variable, variable1])
    plt.show()

data = pd.read_csv("voice.csv")

y = pd.get_dummies(data.label)
features = data.drop(["label"], axis=1)
x = (features - np.min(features))/(np.max(features)-np.min(features)).values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

FEATURES=x.shape[1]


model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=FEATURES))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=100, callbacks=[ModelCheckpoint('bestmodel.h5', save_best_only=True)])

model.load_weights('bestmodel.h5')
model.evaluate(x_test, y_test)

plot(history.history, "accuracy", "val_accuracy")
plot(history.history, "loss", "val_loss")
