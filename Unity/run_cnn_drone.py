from unityagents import UnityEnvironment
import numpy as np
from keras.models import load_model

iters = 4

env = UnityEnvironment(file_name="drone_sim_external", worker_id=0)

model = load_model('cnn_model.h5')

threshold = 10

for i in range(iters):
    print ("ITER", i)
    done = False
    env.reset(train_mode=False)
    states = np.zeros((1, 128, 128, 1))
    while not done:
        action = model.predict(states)
        action = np.hstack((action[0], 0))
        # only move forward
        action[0] = abs(action[0])
        brainInf = env.step(action)['DroneBrain']
        states = brainInf.observations[0]
        # norm = np.linalg.norm(states[0][3:6] - states[0][9:12])
        done = brainInf.local_done[0]
        # done = norm < threshold
