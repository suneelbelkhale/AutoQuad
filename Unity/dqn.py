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
from strategy import RandomArcStrategy, EGreedyArcStrategy
import os, sys
from parser import get_dqn_parser

YAML_FILE = "yamls/linux_dqn.yaml"

actions = [i for i in range(3)] # 0 left 1 straight 2 right
# actions = [np.ones(3) for _ in range(3)]

class DQNAgent(Agent):
    def __init__(self, observation_size, state_size, action_size, max_replay_len=2000):

        super().__init__(observation_size, state_size, action_size, max_replay_len=max_replay_len)
        self.gamma = 0.995    # discount rate
        self.set_epsilons(1.0, 0.999, 0.03)
        self.learning_rate = 0.005
        self._build_model()
        self.target_hard_update_interval = 100
        self.num_train_steps = 0

        self.has_collided = False

    def _build_model(self):
        #processing images

        observation_input = Input(shape=self.observation_size)

        observation_model = Sequential()
        observation_model.add(Conv2D(64, kernel_size=(4, 4), strides=2,
                         activation='relu',
                         input_shape=self.observation_size))
        observation_model.add(Conv2D(64, kernel_size=(4, 4), strides=2,
                         activation='relu'))
        observation_model.add(Conv2D(64, kernel_size=(3, 3), strides=1,
                         activation='relu'))
        observation_model.add(Conv2D(32, kernel_size=(3, 3), strides=1,
                         activation='relu'))
        observation_model.add(Conv2D(16, kernel_size=(3, 3), strides=2,
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
        output = Dense(32, activation='relu')(output)
        output = Dense(self.action_size, activation='linear')(output)

        self.model = Model(inputs=[observation_input, state_input], outputs=output)

        self.model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())


    # def remember(self, state, action, reward, next_state, done):
    #     self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state, observation, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
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

    def compute_reward(self, brainInf, nextBrainInf, action):
        # return reward
        return self.combined_reward_1(brainInf, nextBrainInf, action)


    def combined_reward_1(self, brainInf, nextBrainInf, action):
        dist = brainInf.vector_observations[0][1]
        heading = brainInf.vector_observations[0][0]
        velocity = brainInf.vector_observations[0][2] #forward vel
        collision = nextBrainInf.vector_observations[0][-1] #nextBrainInf.vector_observations[0][-1]
        # 20000 * 0.4^dist, spike reward at the end, but at least differentiable
        dist_reward = 30000 * (0.7**(dist-15))
        # heading_reward: has more influence throughout (thus proportional to distance under 40units away)
        # -32 x^2 + 2 , -45 to 45 degrees is the "positive reward" range
        heading_reward = -16 * heading**2 + 1

        velocity_reward = -100 if abs(velocity) < 0.15 else 0 # step function

        #print("vel: %.2f" % velocity)
        #print("d: %.1f, dr: %.2f  ||  h: %.3f, hr: %.2f" % (dist, dist_reward, heading, heading_reward))
        sys.stdout.flush()
        #self.has_collided = self.has_collided or collision
        if collision:
            print("COLLIDED")
            sys.stdout.flush()
            reward = -20000
        else:
            reward = dist_reward + heading_reward + velocity_reward

        return reward



    def heading_reward(self, brainInf, nextBrainInf, action):
        reward = 0
        #exponential function of dist
        reward += 10.0 / brainInf.vector_observations[0][1]
        collision = nextBrainInf.vector_observations[0][-1]
        # print(collision)
        #maintains state
        self.has_collided = self.has_collided or collision

        goal = int(nextBrainInf.local_done[0] and not self.has_collided)
        reward += collision * -1000
        reward += abs(nextBrainInf.vector_observations[0][0]) * -10.0 # heading diff (normalized -1 to 1 already)
        reward += 20000 * goal

        # if we are done, reset self.has_collided (counts as episode reset)
        if goal:
            print("reached goal")
            sys.stdout.flush()
            self.has_collided = False
        return reward

    def compute_reward_distance(self, brainInf, nextBrainInf, action, done):
        collision = nextBrainInf.vector_observations[0][-1]
        goal = done and not collision
        reward = nextBrainInf.vector_observations[0][1] * -0.1 + collision * -10000 + goal * 10000
        return reward

    def preprocess_observation(self, image):
        return image


if __name__ == "__main__":
    parser = get_dqn_parser()
    parsed_args = parser.parse_args()
    # import ipdb; ipdb.set_trace();
    yaml = YAML_FILE

    #priority to argparser
    # print(parsed_args)
    if parsed_args.yaml:
        yaml = parsed_args.yaml

    params = read_params(yaml)

    #logging stuff, override yaml's entry
    if parsed_args.log_prefix:
        params['logger']['log_prefix'] = parsed_args.log_prefix

    #gpu stuff
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

    # done = False
    #batch_size = params['train']['batch_size']
    #num_episodes = params['train']['num_episodes']
    #max_episode_length = params['train']['max_episode_length']

    exploration_strategy = RandomArcStrategy(action_size=3, args={})

    #runner.run(batch_size=batch_size, num_episodes=num_episodes, max_episode_length=max_episode_length, train_mode=True)
    if parsed_args.type != 2:
        runner = RunAgent(agent, params)
        runner.run(exploration_strategy=exploration_strategy)
    else:
        runner = RunAgent(agent, params, demonstrations=True)
        runner.run_demonstrations()
