from unityagents import UnityEnvironment
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import threading


plt.ion()
# fig, ax = plt.subplots(projection='polar')
ax = plt.subplot(111, projection='polar')
ax.set_thetamin(60)
ax.set_thetamax(120)

# hl, = ax.plot([i for i in range(40)], [i for i in range(40)], 'b-')

max_trajectories = 10
n_bins = 400

freespace = None

env = UnityEnvironment(file_name="SuneelTest_player", worker_id=0)
#print(env)
#print("Success!")

# states = np.load("states.npy")
# actions = np.load("actions.npy")
# print('Num Batches Currently: ', states.shape[0])

def update_plot():
    if freespace is not None:
        plt.cla()

        r = np.linspace(np.pi*2.0/3, np.pi*1.0/3, num=n_bins)
        ax.plot(r,freespace)
        # ax.set_ylim([0,0.1])
        ax.set_thetamin(60)
        ax.set_thetamax(120)
        plt.pause(0.001)
        plt.draw()

# def plot_thr():
#     anim.FuncAnimation(fig, update_plot, frames=100, repeat=True)
#     plt.show()

# thr = threading.Thread(target=plot_thr)
# thr.start()

for i in range(max_trajectories):
    pause = input("press enter to start")
    done = False
    env.reset(train_mode=False)
    episode_states = []
    episode_actions = []

    while not done:
        brainInf = env.step()['DroneBrain']
        # ob = brainInf.observations
        obs = brainInf.vector_observations

        freespace = obs[0,5:]
        # hl.set_xdata([i for i in range(freespace.shape[0])])
        # hl.set_ydata(freespace)
        update_plot()
        print(freespace)

        # states = brainInf.states
        # actions = brainInf.previous_actions[0][:2] #not saving the yaw
        done = brainInf.local_done[0]
        # norm = np.linalg.norm(states[0][3:6] - states[0][9:12])
        # done = norm < threshold
        # import ipdb; ipdb.set_trace();
    # save  = input('Save this trajectory(y/n): ') == 'y'
    # if save:
    #     states_taken.extend(episode_states)
    #     actions_taken.extend(episode_actions)
    end = input('Stop collecting Data(y/n): ') == 'y'
    if end:
        break

# states_taken.extend(np.ndarray.tolist(states))
# actions_taken.extend(np.ndarray.tolist(actions))
# states_taken = np.array(states_taken)
# actions_taken = np.array(actions_taken)
# np.save("states", states_taken)
# np.save("actions", actions_taken)
# print('Num Batches After Data Collection: ', states_taken.shape[0])