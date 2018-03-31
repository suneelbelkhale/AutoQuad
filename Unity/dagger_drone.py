import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from unityagents import UnityEnvironment

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
model.add(Dense(2))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error'])

num_epochs = 300

batch_size = 128
model.fit(states_train,
          actions_train,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=num_epochs,
          shuffle=True)
print()
print(model.metrics_names)
print("Test error:", model.evaluate(states_test, actions_test))

env = UnityEnvironment(file_name="drone_sim_external", worker_id=0)
num_dagger_iterations = 10
steps = 1000
for iterations in range(num_dagger_iterations):
    done = False
    env.reset(train_mode=False)
    states = np.zeros((1, 13))
    threshold = 10
    for i in range(steps):
        action = model.predict(states)
        action = np.hstack((action[0], 0))
        brainInf = env.step(action)['DroneBrain']
        states = brainInf.states
        norm = np.linalg.norm(states[0][3:6] - states[0][9:12])

        #TODO: FIGURE OUT HOW TO GENERATE LABELED ACTIONS
model.save("trained_model.h5")
