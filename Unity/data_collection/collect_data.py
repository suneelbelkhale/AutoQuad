from unityagents import UnityEnvironment
import time
import numpy as np
import argparse
import datetime
from datetime import date

def if_save(collection, new, do_if):
    if do_if:
        if collection is None:
            return new
        else:
            return np.append(collection, new, axis=0)
    return None

def data_collection(args):
    env = UnityEnvironment(file_name=args.env, worker_id=0)
    #print(env)
    #print("Success!")
    print("-- Beginning Data Collection --")

    # frequency = 1
    # states_taken = []
    # actions_taken = []
    # threshold = 10

    obs = None
    states = None
    actions = None
    dones = None

    print("CONFIG: " + str(args.config))

    num_samples_added = 0

    # states = np.load("states.npy")
    # actions = np.load("actions.npy")
    # print('Num Batches Currently: ', states.shape[0])
    for i in range(args.max_trajectories):
        if args.prompt_start_end:
            input("press enter to start")
        done = np.array([[False]])
        env.reset(train_mode=False)

        episode_states = None
        episode_obs = None
        episode_actions = None
        episode_dones = None

        print("-- Beginning Episode %d --" % i)

        ep_num_samples = 0
        # ROLLOUT
        while not done[0][0]:
            brainInf = env.step()['DroneBrain']
            # 1 x 405
            st = brainInf.vector_observations
            # 1 x 128 x 128 x 1
            ob = brainInf.visual_observations[0][:1, :] #camera 1
            # 1 x 1
            ac = brainInf.previous_vector_actions
            # 1 x 1
            done = np.array([[brainInf.local_done[0]]])

            episode_obs = if_save(episode_obs, ob, 'o' in args.config)
            episode_states = if_save(episode_states, st, 's' in args.config)
            episode_actions = if_save(episode_actions, ac, 'a' in args.config)
            episode_dones = if_save(episode_dones, done, 'd' in args.config)

            ep_num_samples += 1
            # episode_states.append(states[0])
            # episode_actions.append(actions)

        if not args.no_prompt_save:
            save  = input('Save this trajectory(y/n): ') == 'y'
        else:
            save = True

        if save:
            obs = if_save(obs, episode_obs, 'o' in args.config)
            states = if_save(states, episode_states, 's' in args.config)
            actions = if_save(actions, episode_actions, 'a' in args.config)
            dones = if_save(dones, episode_dones, 'd' in args.config)

        if args.prompt_start_end:
            end = input('Stop collecting Data(y/n): ') == 'y'
            if end:
                break

        num_samples_added += ep_num_samples

        print("-- Episode %d Terminated w/ %d new samples, %d total  --" % (i, ep_num_samples, num_samples_added))


    print("-- Data Collection Terminated w/ %d samples --" % (num_samples_added))

    # states_taken.extend(np.ndarray.tolist(states))
    # actions_taken.extend(np.ndarray.tolist(actions))
    # states_taken = np.array(states_taken)
    # actions_taken = np.array(actions_taken)
    # np.save("states", states_taken)
    # np.save("actions", actions_taken)
    # print('Num Batches After Data Collection: ', states_taken.shape[0])

    time = datetime.datetime.now().strftime("%H_%M-%b_%d_%y")
    output_name_pref = args.output_dir + args.prefix.replace('.', time)

    obs_to_save = obs
    states_to_save = states
    actions_to_save = actions
    dones_to_save = dones

    if 'o' in args.config:
        if args.input_obs and args.input_obs:
            obs_old = np.load(args.input_obs)
            obs_to_save = np.append(obs_old, obs_to_save, axis=0)
        
        if not args.new_files and args.input_obs:
            np.save(args.input_obs, obs_to_save)
        else:
            np.save(output_name_pref + "_obs", obs_to_save)

    if 's' in args.config:
        if args.input_state:
            states_old = np.load(args.input_state)
            states_to_save = np.append(states_old, states_to_save, axis=0)
        
        if not args.new_files and args.input_state:
            np.save(args.input_state, states_to_save)
        else:
            np.save(output_name_pref + "_states", states_to_save)

    if 'a' in args.config:
        if args.input_act:
            actions_old = np.load(args.input_act)
            actions_to_save = np.append(actions_old, actions_to_save, axis=0)
        
        if not args.new_files and args.input_act:
            np.save(args.input_act, actions_to_save)
        else:
            np.save(output_name_pref + "_actions", actions_to_save)

    if 'd' in args.config:
        if args.input_done:
            dones_old = np.load(args.input_done)
            dones_to_save = np.append(dones_old, dones_to_save, axis=0)
        
        if not args.new_files and args.input_done:
            np.save(args.input_done, dones_to_save)
        else:
            np.save(output_name_pref + "_dones", dones_to_save)



if __name__ == '__main__':
    acceptable_config_types = ['o','s','a','d']
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment to run", type=str, default='drone_sim_player')
    parser.add_argument("--max_trajectories", help="num trajectories to run", type=int, default=1)
    parser.add_argument("--input_obs", help="enter in the np obs file from previously stored runs", type=str, default='')
    parser.add_argument("--input_act", help="enter in the np act file from previously stored runs", type=str, default='')
    parser.add_argument("--input_state", help="enter in the np state file from previously stored runs", type=str, default='')
    parser.add_argument("--input_done", help="enter in the np input file from previously stored runs", type=str, default='')
    parser.add_argument("--config", help="some period separated combination of " +
                            "\n\t o (visual observations), s (vector observations), a (actions), d (dones)" +
                            "\n\t e.g. 'o.s.a.d' ", type=str, default='o.s.a.d')
    parser.add_argument("--new_files", help="true if you want to not append everything", action='store_true')
    parser.add_argument("--prefix", help="this prepends _obs, _actions, etc, where . can be used to denote a time input (e.g. --prefix 'dsim_.'", type=str, default='sim_.')
    parser.add_argument("--no_prompt_save", help="do this to turn off prompting for saving trajectories", action='store_true')
    parser.add_argument("--prompt_start_end", help="do this to turn off prompting for saving trajectories", action='store_true')
    parser.add_argument("--output_dir", help="where to output data with ending /", type=str, default='')
    args = parser.parse_args()
    args.config = [el.strip() for el in str.split(args.config, '.') if (el.strip() in acceptable_config_types)]
    
    data_collection(args)


