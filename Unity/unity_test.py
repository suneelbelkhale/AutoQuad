from unityagents import UnityEnvironment
import time

env = UnityEnvironment(file_name="drone_sim_windows.exe", worker_id=0)
print(env)
env.reset(train_mode=False)
print("Success!")
done = False
## TODO
# timing
# when to call done
# print out non zero actions
while not done:
    brainInf = env.step()['DroneBrain']
    ob = brainInf.observations
    states = brainInf.states
    actions = brainInf.previous_actions[0]
    done = brainInf.local_done[-1]
    print(states)
    # if actions.any():
    #     print("NON ZERO ACTION: ", actions)

env.close()
# print(ob[0].shape)
# print(states)
# print(done)
# print(brainInf)
# print("--------------------------")
