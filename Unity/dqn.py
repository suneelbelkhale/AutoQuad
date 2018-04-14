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
actions = [i for i in range(3)] # 0 left 1 straight 2 right
# actions = [np.ones(3) for _ in range(3)]

class DQNAgent(Agent):
    def __init__(self, state_size, action_size):
        # self.state_size = state_size
        # self.action_size = action_size
        # self.replay_buffer = deque(maxlen=2000)
        super().__init__(state_size, action_size)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self._build_model()
        self.target_hard_update_interval = 100
        self.num_train_steps = 0

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
        for so1, action, reward, so2, done in minibatch:
            
            state, obs = so1
            next_state, next_obs = so2

            target_vector = self.target_model.predict(next_obs)

            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.max(target_vector[0]))
            target_vector[0][action] = target

            self.model.fit(obs, target_vector, epochs=1, verbose=1)
            self.num_train_steps += 1
            if self.num_train_steps % self.target_hard_update_interval==0:
                self.target_model = keras.models.clone_model(self.model)
                self.target_model.set_weights(self.model.get_weights())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def compute_reward(self, brainInf, nextBrainInf, action):
        reward = nextBrainInf.states[0][1]
        return reward

    def preprocess_observation(self, image):
        return image

if __name__ == "__main__":


    # env = UnityEnvironment(file_name="drone_sim_external", worker_id=0)
    state_size = (128, 128, 1)
    action_size = 3
    agent = DQNAgent(state_size, action_size)

    runner = RunAgent(agent, "drone_sim_external")

    # done = False
    batch_size = 32
    num_episodes = 1000

    runner.run(batch_size=batch_size,num_episodes=num_episodes)

    # for e in range(num_episodes):
    #     state = env.reset(train_mode=False)
    #     state = state['DroneBrain'].observations
    #     state = state[0]
    #     for time in range(500):
    #         action = agent.act(state)
    #         brainInf = env.step(action)['DroneBrain']
    #         true_states = brainInf.states[0]
    #         reward = true_states[1]
    #         next_state = agent.preprocess(brainInf.observations[0])
    #         done = brainInf.local_done
    #         collided = true_states[4] == 1
    #         reward = reward if not collided else -200
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         if len(agent.replay_buffer) > batch_size:
    #             agent.train(batch_size)
    #         if done:
    #             print("episode: {}/{}, score: {}, e: {:.2}"
    #                   .format(e, num_episodes, time, agent.epsilon))
    #             break
