# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from run_model import RunAgent
from agent import Agent
import keras
from yaml_loader import read_params

import os


actions = [i for i in range(3)] # 0 left 1 straight 2 right
# actions = [np.ones(3) for _ in range(3)]

class DQNAgent(Agent):
    def __init__(self, state_size, action_size, max_replay_len=2000):

        super().__init__(state_size, action_size, max_replay_len=max_replay_len)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self._build_model()
        self.target_hard_update_interval = 100
        self.num_train_steps = 0
        self.state_size = state_size

    def _build_model(self):
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
        self.model = model
        self.target_model = keras.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())

    # def remember(self, state, action, reward, next_state, done):
    #     self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state, observation):
        if np.random.rand() <= self.epsilon:
            return actions[random.randrange(self.action_size)]
        act_values = self.model.predict(observation)
        return actions[np.argmax(act_values[0])] # returns action

    def train(self, batch_size):
        minibatch = self.sample(batch_size)
        observations = np.zeros(([batch_size]+list(self.state_size)))
        next_observations = np.zeros(([batch_size]+list(self.state_size)))
        actions = np.zeros(batch_size)
        rewards = np.zeros(batch_size)
        dones = np.zeros(batch_size)
        for i, (so1, action, reward, so2, done) in enumerate(minibatch):
            state, obs = so1
            next_state, next_obs = so2
            observations[i] = obs
            next_observations[i] = next_obs
            actions[i] = action
            rewards[i] = reward
            if done == True:
                dones[i] = 1
            else:
                dones[i] = 0
        targets = self.target_model.predict(next_observations)
        target = np.where(dones == 0, rewards + self.gamma * np.max(targets, axis=1), rewards)
        for i in range(batch_size):
            targets[i][actions.astype(int)[i]] = target[i]
        self.model.fit(observations, targets, epochs=1, verbose=1)
        self.num_train_steps += 1
        if self.num_train_steps % self.target_hard_update_interval==0:
            self.target_model = keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def compute_reward(self, brainInf, nextBrainInf, action):
        reward = nextBrainInf.vector_observations[0][1] * -1 + nextBrainInf.vector_observations[0][-1] * -1000
        return reward

    def preprocess_observation(self, image):
        return image

if __name__ == "__main__":

    params = read_params("yamls/mac_dqn.yaml")

    if params['gpu']['device'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(params['gpu']['device'])

    # env = UnityEnvironment(file_name="drone_sim_external", worker_id=0)
    state_size = (128, 128, 1)
    action_size = 3
    max_replay_len = params['train']['max_replay_len']
    agent = DQNAgent(state_size, action_size, max_replay_len=max_replay_len)

    runner = RunAgent(agent, params)

    # done = False
    #batch_size = params['train']['batch_size']
    #num_episodes = params['train']['num_episodes']
    #max_episode_length = params['train']['max_episode_length']

    #runner.run(batch_size=batch_size, num_episodes=num_episodes, max_episode_length=max_episode_length, train_mode=True)
    runner.run()
