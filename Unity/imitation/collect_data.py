from unityagents import UnityEnvironment
import time
import numpy as np

env = UnityEnvironment(file_name="drone_sim_player", worker_id=0)
#print(env)
#print("Success!")

max_trajectories = 100
frequency = 1
states_taken = []
actions_taken = []
threshold = 10

# states = np.load("states.npy")
# actions = np.load("actions.npy")
# print('Num Batches Currently: ', states.shape[0])
for i in range(max_trajectories):
    pause = input("press enter to start")
    done = False
    env.reset(train_mode=False)
    episode_states = []
    episode_actions = []
    while not done:
        brainInf = env.step()['DroneBrain']
        ob = brainInf.observations
        states = brainInf.states
        actions = brainInf.previous_actions[0][:2] #not saving the yaw
        norm = np.linalg.norm(states[0][3:6] - states[0][9:12])
        done = norm < threshold
        episode_states.append(states[0])
        episode_actions.append(actions)
    save  = input('Save this trajectory(y/n): ') == 'y'
    if save:
        states_taken.extend(episode_states)
        actions_taken.extend(episode_actions)
    end = input('Stop collecting Data(y/n): ') == 'y'
    if end:
        break
# states_taken.extend(np.ndarray.tolist(states))
# actions_taken.extend(np.ndarray.tolist(actions))
states_taken = np.array(states_taken)
actions_taken = np.array(actions_taken)
np.save("states", states_taken)
np.save("actions", actions_taken)
# print('Num Batches After Data Collection: ', states_taken.shape[0])