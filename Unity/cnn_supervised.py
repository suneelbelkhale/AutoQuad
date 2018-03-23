import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import Adam

#load data
from sklearn.model_selection import train_test_split
seed=666
np.random.seed(seed)
obs = np.load("images.npy")
states = np.load("states.npy")
actions = np.load("actions.npy")

obs = np.array([obs[i][0] for i in range(len(obs))])


states_train, states_test, actions_train, actions_test = \
    train_test_split(obs, actions, test_size=0.33, random_state=seed)


print(states_train.shape)
#create model
model = Sequential()
model.add(Conv2D(input_shape=(128,128,1), filters=4, kernel_size=7, strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))
model.add(Conv2D(16, kernel_size=3, strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))
model.add(Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))
model.add(Conv2D(16, kernel_size=3, strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4), strides=None, padding='same'))
model.add(Conv2D(8, kernel_size=3, strides=(1, 1), padding='same', activation='relu'))

model.add(Reshape((4*4*8,)))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error'])

print(model.summary())

num_epochs = 80

batch_size = 32
model.fit(states_train,
          actions_train,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=num_epochs,
          shuffle=True)
print()
print(model.metrics_names)
print("Test error:", model.evaluate(states_test, actions_test))

model.save("cnn_model.h5")