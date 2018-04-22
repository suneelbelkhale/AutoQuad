# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.optimizers import Adam
from run_model import RunAgent
from agent import Agent
import keras
from yaml_loader import read_params

import os


actions = [i for i in range(3)] # 0 left 1 straight 2 right
# actions = [np.ones(3) for _ in range(3)]

class DQNAgent(Agent):
    def __init__(self, observation_size, state_size, action_size, max_replay_len=2000):

        super().__init__(observation_size, state_size, action_size, max_replay_len=max_replay_len)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self._build_model()
        self.target_hard_update_interval = 100
        self.num_train_steps = 0

    def _build_model(self):

        #processing images

        observation_input = Input(shape=self.observation_size)

        observation_model = Sequential()
        observation_model.add(Conv2D(64, kernel_size=(4, 4), strides=2,
                         activation='relu',
                         input_shape=self.observation_size))
        observation_model.add(Conv2D(64, kernel_size=(4, 4), strides=2,
                         activation='relu'))
        observation_model.add(Conv2D(32, kernel_size=(3, 3), strides=1,
                         activation='relu'))
        observation_model.add(Flatten())
        observation_model.add(Dense(128, activation='relu'))
        observation_model.add(Dense(16, activation='relu'))
        
        #processing state

        state_input = Input(shape=self.state_size)

        state_model = Sequential()
        state_model.add(Dense(32, activation='relu', input_shape=self.state_size))
        state_model.add(Dense(8, activation='relu'))

        #combining

        concatenated = keras.layers.concatenate([observation_model(observation_input), state_model(state_input)])
        output = Dense(64, activation='relu')(concatenated)
        output = Dense(self.action_size, activation='linear')(output)

        self.model = Model(inputs=[observation_input, state_input], outputs=output)

        self.model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())


    # def remember(self, state, action, reward, next_state, done):
    #     self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state, observation):
        if np.random.rand() <= self.epsilon:
            return actions[random.randrange(self.action_size)]
        act_values = self.model.predict([observation, state])
        return actions[np.argmax(act_values[0])] # returns action

    def train(self, batch_size):
        minibatch = self.sample(batch_size)

        observations = np.zeros(([batch_size]+list(self.observation_size)))
        states = np.zeros(([batch_size]+list(self.state_size)))
        next_observations = np.zeros(([batch_size]+list(self.observation_size)))
        next_states = np.zeros(([batch_size]+list(self.state_size)))

        actions = np.zeros(batch_size)
        rewards = np.zeros(batch_size)
        dones = np.zeros(batch_size)

        for i, (so1, action, reward, so2, done) in enumerate(minibatch):
            state, obs = so1
            next_state, next_obs = so2
            #print(next_state.shape)
            observations[i] = obs
            states[i] = state
            next_observations[i] = next_obs
            next_states[i] = next_state
            actions[i] = action
            rewards[i] = reward
            if done == True:
                dones[i] = 1
            else:
                dones[i] = 0
        targets = self.target_model.predict([next_observations, next_states])
        target = np.where(dones == 0, rewards + self.gamma * np.max(targets, axis=1), rewards)
        for i in range(batch_size):
            targets[i][actions.astype(int)[i]] = target[i]
        self.model.fit([observations, states], targets, epochs=1, verbose=0)
        self.num_train_steps += 1
        if self.num_train_steps % self.target_hard_update_interval==0:
            self.target_model = keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def compute_reward(self, brainInf, nextBrainInf, action):
        # return reward
        return self.heading_reward(brainInf, nextBrainInf, action)

    def heading_reward(self, brainInf, nextBrainInf, action):
        reward = 0
        #exponential function of dist
        collision = nextBrainInf.vector_observations[0][-1]
        print(collision)
        goal = nextBrainInf.local_done[0] and not collision
        if goal:
        	print("reached goal")
        reward += collision * -1000
        reward += abs(nextBrainInf.vector_observations[0][0]) * -0.1 # heading diff (normalized -1 to 1 already)
        reward += 5000 * goal
        return reward

    def compute_reward_distance(self, brainInf, nextBrainInf, action, done):
        collision = nextBrainInf.vector_observations[0][-1]
        goal = done and not collision
        reward = nextBrainInf.vector_observations[0][1] * -0.1 + collision * -10000 + goal * 10000
        return reward

    def preprocess_observation(self, image):
        return image

if __name__ == "__main__":

    params = read_params("yamls/linux_dqn.yaml")

    if params['gpu']['device'] >= 0:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        #from tensorflow.python.client import device_lib
        #print(device_lib.list_local_devices())
        os.environ["CUDA_VISIBLE_DEVICES"]=str(params['gpu']['device'])

    # env = UnityEnvironment(file_name="drone_sim_external", worker_id=0)
    observation_size = (128, 128, 1)
    state_size = (5,)
    action_size = 3
    max_replay_len = params['train']['max_replay_len']
    agent = DQNAgent(observation_size, state_size, action_size, max_replay_len=max_replay_len)

    runner = RunAgent(agent, params)

    # done = False
    #batch_size = params['train']['batch_size']
    #num_episodes = params['train']['num_episodes']
    #max_episode_length = params['train']['max_episode_length']

    #runner.run(batch_size=batch_size, num_episodes=num_episodes, max_episode_length=max_episode_length, train_mode=True)
    runner.run()
