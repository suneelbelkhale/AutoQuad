import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam

#load data
states = np.load("states.npy")
actions = np.load("actions.npy")

#create model
model = Sequential()
model.add(Dense(128, input_shape=(13,)))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(3))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error'])

num_epochs = 300

batch_size = 128
model.fit(states, actions,
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True)
print()
print(model.metrics_names)
print("Train error:", model.evaluate(states, actions))

model.save("trained_model.h5")