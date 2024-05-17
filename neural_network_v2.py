import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float) 
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

celsius_test = np.array([-15, 20, 40], dtype=float)
fahrenheit_test = np.array([5, 68, 104], dtype=float)

# Normalización de datos
celsius = (celsius - celsius.min())/(celsius.max() - celsius.min())
celsius_test = (celsius_test - celsius_test.min())/(celsius_test.max() - celsius_test.min())

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=[1])) 
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

opt = Adam(learning_rate=0.01)
model.compile(loss='mse', metrics=[MeanAbsoluteError()], optimizer=opt)

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

model.fit(celsius, fahrenheit, epochs=600, validation_data=(celsius_test, fahrenheit_test), 
            callbacks=[early_stop], verbose=0)

print(model.evaluate(celsius_test, fahrenheit_test))

result = model.predict([100.0]) 
print("El resultado es " + str(result) + " fahrenheit!")