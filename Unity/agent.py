## super simple Agent template
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import keras

class Agent:
    def __init__(self, observation_size, state_size, action_size, max_replay_len=2000):
        self.observation_size = observation_size
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=max_replay_len)
        self.model = Sequential()
        self.set_epsilons(0.0, 0.0, 0.0)

    #processes and returns an observation, can also handle compression, buffer stores the output of this
    # OVERRIDE THIS
    def preprocess_observation(self, observation):
        return observation

    #returns action, assume obs is preprocessed already
    #set greedy False = use epsilon greedy
    #set training False = inference
    # OVERRIDE THIS
    def act(self, state, observation, greedy=False, training=True):
        return 0

    # OVERRIDE THIS
    def train(self, batch_size):
        pass

    # OVERRIDE THIS
    def compute_reward(self, currBrainInf, nextBrainInf, action):
        return 0

    def sample(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    #stores a sample ((s,o),a,r,(s',o'),d)
    # OVERRIDE for different sample length / sample parsing
    def store_sample(self, sample):
        if len(sample) == 5:
            self.replay_buffer.append(sample)
        else:
            print("Cannot store sample -- incorrect length")

    #returns current epsilon, and decay rate in a tuple
    def get_epsilons(self):
        return self.epsilon, self.epsilon_decay, self.epsilon_min

    def set_epsilons(self, epsilon, epsilon_decay, epsilon_min):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
