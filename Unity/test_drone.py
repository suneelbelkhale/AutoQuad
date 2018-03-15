from unityagents import UnityEnvironment
import numpy as np
from keras.models import load_model

env = UnityEnvironment(file_name="drone_sim_external", worker_id=0)

model = load_model('trained_model.h5')
done = False
env.reset(train_mode=False)
states = np.zeros((1, 13))
threshold = 10
while not done:
    action = model.predict(states)
    action = np.hstack((action[0], 0))
    brainInf = env.step(action)['DroneBrain']
    states = brainInf.states
    norm = np.linalg.norm(states[0][3:6] - states[0][9:12])
    done = norm < threshold
