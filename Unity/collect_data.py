from unityagents import UnityEnvironment
import time
import numpy as np

env = UnityEnvironment(file_name="drone_sim_player", worker_id=0)
#print(env)
#print("Success!")

load = False

max_trajectories = 100
frequency = 1
obs_taken = []
states_taken = []
actions_taken = []

if load:
    try:     
        obs_taken = np.load("images", obs_taken)
        states_taken = np.load("states", states_taken)
        actions_taken = np.load("actions", actions_taken)
    except Exception as e:
        print("Loading numpy arrays failed", str(e))

threshold = 10

for i in range(max_trajectories):
    pause = input("press enter to start")
    done = False
    env.reset(train_mode=False)
    count = 0
    episode_states = []
    episode_actions = [] 
    episode_obs = [] #images
    while not done:
        count = (count + 1)
        brainInf = env.step()['DroneBrain']
        ob = brainInf.observations
        states = brainInf.states
        actions = brainInf.previous_actions[0][:2] #not saving the yaw
        norm = np.linalg.norm(states[0][3:6] - states[0][9:12])
        done = norm < threshold
        if count % frequency == 0:
            episode_obs.append(ob[0])
            episode_states.append(states[0])
            episode_actions.append(actions)
    save = input('Save this trajectory(y/n): ') == 'y'
    if save:
        obs_taken.extend(episode_obs)
        states_taken.extend(episode_states)
        actions_taken.extend(episode_actions)
    end = input('Stop collecting Data(y/n): ') == 'y'
    if end:
        break
obs_taken = np.array(obs_taken)
states_taken = np.array(states_taken)
actions_taken = np.array(actions_taken)
np.save("images", obs_taken)
np.save("states", states_taken)
np.save("actions", actions_taken)
env.close()
    # print(ob[0].shape)
    # print(states)
    # print(done)
    # print(brainInf)
    # print("--------------------------")
