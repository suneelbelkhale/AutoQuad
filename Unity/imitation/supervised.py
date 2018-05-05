import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam

#load data
from sklearn.model_selection import train_test_split
seed=666
np.random.seed(seed)
states = np.load("states.npy")
actions = np.load("actions.npy")

states_train, states_test, actions_train, actions_test = \
    train_test_split(states, actions, test_size=0.33, random_state=seed)
#create model
model = Sequential()
model.add(Dense(128, input_shape=(13,)))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(2))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-3),
              metrics=['mean_squared_error'])

num_epochs = 300

batch_size = 256
model.fit(states_train,
          actions_train,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=num_epochs,
          shuffle=True)
print()
print(model.metrics_names)
print("Test error:", model.evaluate(states_test, actions_test))

model.save("trained_model.h5")