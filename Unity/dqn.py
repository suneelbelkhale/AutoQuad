# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from unityagents import UnityEnvironment

actions = [np.ones(3)]
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate))
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(4, 4), strides=2,
                         activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=2,
                         activation='relu'))
        model.add(Conv2D(32, kernel_size=(3, 3), strides=1,
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return actions[random.randrange(self.action_size)]
        act_values = self.model.predict(state)
        return actions[np.argmax(act_values[0])] # returns action

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = UnityEnvironment(file_name="drone_sim_external", worker_id=0)
    state_size = (13, 1)
    action_size = 1
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    num_episodes = 1000
    for e in range(num_episodes):
        state = env.reset(train_mode=False)
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            brainInf = env.step(action)['DroneBrain']
            #TODO: get next state, reward, terminal from brainInf
            next_state = brainInf.states
            reward = brainInf.reward
            done = brainInf.local_done
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, num_episodes, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

#TODO:
'''
- setup images 
- set target network to have delayed parameters
- 
'''