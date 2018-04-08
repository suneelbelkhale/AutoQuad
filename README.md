# Project AutoQuad
<div style="text-align: left; display: inline-block">
    <img src="https://uav.berkeley.edu/wp-content/uploads/2017/09/logo_full_text_light-1.png" width="30%">
    <img src="https://ml.berkeley.edu/decals/DSD/images/logo.png", height="100px">
</div>
#### Code Base for Project AutoQuad, Spring 2018 
A joint project between ML@Berkeley and UAVs@Berkeley.

---
### Setup Instructions

1. Run `pip install unityagents` to get the mlagents python code
2. Run `pip install numpy h5py tensorflow` (if you have a GPU, use `tensorflow-gpu`) 
3. Build the Unity Drone environment into a binary (<a href="https://github.com/UAVs-at-Berkeley/UnityDroneSim">UnityDroneSim repo</a>)
    1. Make sure that the Brain Mode attribute (look in the inspector) in the DroneBrain Game Object (under the DroneAcademy Game Object) is set to Player Mode (if you want to collect samples) and External Mode (if you want to run a trained model). If you are doing this for the first time, build one executable in each mode. You will need to rebuild every time you change the simulator.
    2. TO BUILD: File -> Build Settings, Build, and then save the binary/exe to the Unity/ folder of this repository (or somewhere in your python path). Save it as "drone_sim_player" (for player mode) or "drone_sim_external" (for external mode).
    3. See <a href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md">here</a> under the "Building Unity Environment" section for building help.
4. Follow <a href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Unity-Agents---Python-API.md">this</a> tutorial or run one of the python scripts in the Unity folder.

---
### Files

#### Imitation Learning (Just state)

_Currently in branch master_

collect_data.py
* As of now, no arguments
* Runs Player mode until collision, then resets 
* Allows to individually discard bad trajectories
* Produces 3 numpy arrays: actions.npy, states.npy, and images.npy (used in supervised)
* To not save images, comment out this line: `np.save("images", obs_taken)`

supervised.py
* As of now, no arguments
* Basic neural network implementation (Keras), MSE Loss
* Change `num_epochs` for more epochs or `batch_size` for larger batches.
* saves "trained_model.h5"

test_drone.py
* As of now, no arguments
* Runs `iters` number of rollouts using the trained_model.h5 from supervised.py on the "drone_sim_external" unity game.
* Make sure `train_mode=False` in `env.reset` in order for realtime visualization of the game.

#### Imitation Learning (State + Images)

_Currently in branch master_

collect_data.py 
* Make sure all 3 np arrays are being saved here. See above for more info.

cnn_supervised.py
* As of now, no arguments, but requires all 3 npy files.
* Basic convolutional neural network (Keras), MSE Loss
* Change `num_epochs` for more epochs or `batch_size` for larger batches.
* saves "cnn_model.h5"

run_cnn_drone.py
* As of now, no arguments
* Similar to test_drone.py but runs cnn_model.h5

#### Q-Learning

In progress.

---
### Members

#### ML@Berkeley:  
*Project Managers:* Suneel Belkhale, Alex Li  
*Project Members:* Gefen Kohavi, Murtaza Dalal, Daniel Ho, Franklin Rice, Allan Levy

##### UAV's@Berkeley:  
*Project Managers:* Suneel Belkhale  
*Project Members:* Alex Chan, Kevin Yang, Valmik Prabhu, Isabella Maceda, Erin Song, Dilan Bhalla, Asad Abbasi, Jason Kim

---
### References

N/A