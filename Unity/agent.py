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

    #processes and returns an observation, can also handle compression, buffer stores the output of this
    # OVERRIDE THIS
    def preprocess_observation(self, observation):
        return observation

    #returns action, assume obs is preprocessed already
    # OVERRIDE THIS
    def act(self, state, observation):
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

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)