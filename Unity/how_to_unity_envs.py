# from unityagents import UnityEnvironment
from mlagents.envs import UnityEnvironment

import numpy as np
from keras.models import load_model

import argparse


def main(args):
    env = UnityEnvironment(file_name=args.env, worker_id=0)

    model = load_model(args.model)
    done = False
    env.reset(train_mode=False)
    states = np.zeros((1, 13))
    threshold = 10
    while not done:
        action = model.predict(states)
        action = np.hstack((action[0], 0))
        brainInf = env.step(action)['DroneBrain']
        states = brainInf.vector_observations
        # norm = np.linalg.norm(states[0][3:6] - states[0][9:12])
        # done = norm < threshold



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='environment to run', type=str)
    parser.add_argument('model', help='model (h5 file name) to be run', type=str)
    args = parser.parse_args()
    main(args)