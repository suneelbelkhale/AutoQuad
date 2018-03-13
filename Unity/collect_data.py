from unityagents import UnityEnvironment
import time
import numpy as np

env = UnityEnvironment(file_name="drone_sim_player", worker_id=0)
#print(env)
#print("Success!")

num_trajectories = 1
frequency = 4
states_taken = []
actions_taken = []
threshold = 10

for i in range(num_trajectories):
    pause = input("press enter to start")
    done = False
    env.reset(train_mode=False)
    count = 0

    while not done:
        count = (count + 1)
        brainInf = env.step()['DroneBrain']
        ob = brainInf.observations
        states = brainInf.states
        actions = brainInf.previous_actions[0]
        norm = np.linalg.norm(states[0][3:6] - states[0][9:12])
        done = norm < threshold
        if count % frequency == 0:
            states_taken.append(states[0])
            actions_taken.append(actions)

states_taken = np.array(states_taken)
actions_taken = np.array(actions_taken)
np.save("states", states_taken)
np.save("actions", actions_taken)
env.close()
    # print(ob[0].shape)
    # print(states)
    # print(done)
    # print(brainInf)
    # print("--------------------------")
