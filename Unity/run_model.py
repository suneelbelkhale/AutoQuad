from unityagents import UnityEnvironment
import numpy as np
import time
import sys

from unityagents import BrainInfo

from logger import Logger

from yaml_loader import read_params


from strategy import RandomArcStrategy, EGreedyArcStrategy

#utility

# true_print = print
# def print(*args):
#     true_print(*args)
#     sys.stdout.flush()

class RunAgent:

    # args is either params dict or a yaml filename
    def __init__(self, agent, args, demonstrations=False):

        if not isinstance(args, dict):
            #read as yaml file
            args = read_params(args)

        self._agent = agent
        if demonstrations:
            self.env_name = args['system']['demonstration_drone_sim']
        else:
            self.env_name = args['system']['drone_sim']
        self._args = args
        self.create_env(self.env_name)
        self.parseArgs(self._args)

        self.lg = Logger(self.log_prefix, display_name="run", console_lvl=self.console_lvl)

    def parseArgs(self, args):
        #training
        self.max_episode_length = args['train']['max_episode_length']
        self.num_episodes = args['train']['num_episodes']
        self.batch_size = args['train']['batch_size']
        self.train_period = args['train']['train_period']
        self.train_on_demonstrations = args['train']['train_on_demonstrations']
        self.demonstration_eval_episodes = args['train']['demonstration_eval_episodes']
        self.demonstration_epochs = args['train']['demonstration_epochs']
        self.train_after_episode = args['train']['train_after_episode']
        self.reinforce_good_episodes = args['train']['reinforce_good_episodes']
        self.good_ep_thresh = args['train']['good_ep_thresh']
        self.model_file = args['train']['model_file']
        #inference
        self.only_inference = args['inference']['only_inference']
        self.num_inference_episodes = args['inference']['num_inference_episodes']
        #demonstrations

        #logging
        self.log_prefix = args['logging']['log_prefix']
        self.console_lvl = args['logging']['console_lvl']

    def create_env(self, file_name):
        self._env = UnityEnvironment(file_name=file_name, worker_id=1)

    #gives them pretty high reward
    def train_demonstrations(self):
        self.lg.print("Beginning Training on demonstrations", lvl="info")

        states_taken = np.load("demonstrated_states.npz")
        observations_taken = np.load("demonstrated_observations.npz")
        actions_taken = np.load("demonstrated_actions.npz")
        #import ipdb; ipdb.set_trace();
        states_taken = states_taken.f.arr_0
        observations_taken= observations_taken.f.arr_0
        actions_taken = actions_taken.f.arr_0

        total_seen = 0

        for ep in range(len(states_taken)):
            self.lg.print("Training on demonstration:", ep, lvl="info")
            #loop over episode
            for epoch in range(self.demonstration_epochs):
                self.lg.print("EPOCH: %d/%d" % (epoch, self.demonstration_epochs))
                for i in reversed(range(len(states_taken[ep]) - 1)):
                    these_states = states_taken[ep][i]
                    these_observations = observations_taken[ep][i]
                    these_actions = actions_taken[ep][i]

                    next_states = states_taken[ep][i+1]
                    next_observations = observations_taken[ep][i+1]
                    next_actions = actions_taken[ep][i+1]

                    done = i == len(these_states) - 1

                    this_bi = BrainInfo(these_observations, these_states, None)
                    next_bi = BrainInfo(next_observations, next_states, None)

                    this_bi.local_done = [False]
                    next_bi.local_done = [done]

                    reward = self._agent.compute_reward(this_bi, next_bi, these_actions)

                    #artificial boost to reward --> must check for arficial zero issues
                    # (discount ^ numstepsleft) * TERMINAL_REWARD
                    # logically boost the paths that have been shown to you
                    reward = max(reward, 0) + 100

                    sample = ((these_states, these_observations),
                                these_actions,
                                reward,
                                (next_states, next_observations),
                                done
                            )

                    self._agent.store_sample(sample)
                    total_seen += 1

                    if total_seen > self.batch_size*2:
                        if total_seen % (self.train_period) == 0:
                            self._agent.train(batch_size=self.batch_size*2)


                    #these_states = next_states
                    #these_observations = next_observations
                    #these_actions = next_actions

        self.lg.print("Looped over %d demonstrated samples" % total_seen)
        return total_seen > 0

    #must pass in exploration_strategy object for training
    def run(self, load=False, exploration_strategy=None):
        if self.only_inference:
            self.lg.print("-- Running INFERENCE on %d episodes of length %d -- \n" % (self.num_inference_episodes, self.max_episode_length), lvl="info")
            self.run_inference()
        else:
            self.lg.print("-- Running TRAINING on %d episodes of length %d -- \n" % (self.num_episodes, self.max_episode_length), lvl="info")
            self.run_training(load=load,
                              exploration_strategy=exploration_strategy)
 
    #must pass in exploration_strategy object
    def run_training(self, sim_train_mode=True, load=False,
                     exploration_strategy=None): #batch_size=32, num_episodes=1, max_episode_length=1000, train_period=3, train_after_episode=False, train_mode=False):

        if load:
            try:
                self._agent.load(self.model_file)
            except Exception as e:
                self.lg.print("Could not load from file:", str(e), "error")

        if self.train_on_demonstrations:
            success = self.train_demonstrations()
            if success:
                self.dem_ep_count = 0 #basically runs trajectory mostly greedily first

        for e in range(self.num_episodes):
            walltime = time.time()

            #reset
            brainInf = self._env.reset(train_mode=sim_train_mode)['DroneBrain']
            # import ipdb; ipdb.set_trace()

            p_observation = self._agent.preprocess_observation(brainInf.visual_observations[0])

            rewards = []
            done = False

            self.lg.print("-- Episode %d --" % e)
            sys.stdout.flush()

            greedy = e < self.demonstration_eval_episodes

            episode_samples = []

            if exploration_strategy is not None:
                trajectory = exploration_strategy.generate_trajectory(args={})
            else:
                trajectory = None

            for t in range(self.max_episode_length):
                #generalized act function takes in state and observations (images)

                if trajectory is None:
                    action = self._agent.act(brainInf.vector_observations, p_observation, greedy=greedy)
                else:
                    #use exploration strategy
                    action = next(trajectory)

                nextBrainInf = self._env.step(action)['DroneBrain']

                done = brainInf.local_done[0]
                #self.lg.print(brainInf.local_done)
                reward = self._agent.compute_reward(brainInf, nextBrainInf, action)
                rewards.append(reward)

                next_p_observation = self._agent.preprocess_observation(nextBrainInf.visual_observations[0])

                #stores processed things
                sample = (  (brainInf.vector_observations[0], p_observation), 
                            action,
                            reward,
                            (nextBrainInf.vector_observations[0], next_p_observation),
                            done    )

                episode_samples.append(sample)
                self._agent.store_sample(sample)

                #train every experience here
                if not self.train_after_episode and len(self._agent.replay_buffer) > self.batch_size:
                    if t % self.train_period == 0:
                        self._agent.train(self.batch_size)

                if t % 100 == 0:
                    #self.lg.print("step", t)
                    sys.stdout.flush()

                if done:
                    break

                p_observation = next_p_observation
                brainInf = nextBrainInf

            #LOGGING
            episode_str = ( "Episode {}/{} completed,"
                           " \n\t total steps: {}," 
                           " \n\t total reward: {},"
                           " \n\t mean reward: {},"
                           " \n\t max reward: {},"
                           " \n\t min reward: {},"
                           " \n\t greedy: {},"
                           " \n\t epsilon: {},"
                           " \n\t sim time: {}" )
            self.lg.print(episode_str.format(e, self.num_episodes, t, np.sum(rewards),
                  np.mean(rewards), np.max(rewards), np.min(rewards), greedy,
                             self._agent.epsilon, time.time() - walltime), lvl="info")
            # train after episode
            if self.train_after_episode and len(self._agent.replay_buffer) > self.batch_size:
                walltime = time.time()
                for i in range(t // self.train_period):
                    self._agent.train(self.batch_size)
                self.lg.print("training complete in: {}".format(time.time() - walltime))
                sys.stdout.flush()

            #good episode reinforcement
            if np.sum(rewards) >= self.good_ep_thresh:
                print("-- SUCCESSFUL EPISODE. Training More -- ")
                new_batch_size = min(len(self._agent.replay_buffer), self.batch_size * self.reinforce_good_episodes)
                self._agent.train(new_batch_size)

            # save after an episode
            if e % 10 == 0:
                self._agent.save(self.model_file)

            # update epsilon after each episode
            self._agent.epsilon_update()

        self.lg.print("|------------| TRAINING COMPLETE |------------|", lvl="info")

    def run_inference(self, train_mode=False):
        self._agent.load(self.model_file)

        #reset
        for e in range(self.num_inference_episodes):
            walltime = time.time()

            #reset
            brainInf = self._env.reset(train_mode=train_mode)['DroneBrain']
            # import ipdb; ipdb.set_trace()

            p_observation = self._agent.preprocess_observation(brainInf.visual_observations[0])

            rewards = []
            done = False

            self.lg.print("-- Episode %d --" % e)

            for t in range(self.max_episode_length):
                #generalized act function takes in state and observations (images)

                action = self._agent.act(brainInf.vector_observations,
                                         p_observation, greedy=True)

                nextBrainInf = self._env.step(action)['DroneBrain']

                done = brainInf.local_done[0]
                # self.lg.print(brainInf.local_done)
                reward = self._agent.compute_reward(brainInf, nextBrainInf, action)
                rewards.append(reward)

                if done:
                   break

                brainInf = nextBrainInf

            #LOGGING
            episode_str = ( "Episode {}/{} completed,"
                           " \n\t total steps: {}," 
                           " \n\t total reward: {},"
                           " \n\t mean reward: {},"
                           " \n\t max reward: {},"
                           " \n\t min reward: {},"
                           " \n\t epsilon: {},"
                           " \n\t sim time: {}" )
            # self.lg.print("Episode {}/{} completed, \n\t total steps: {},\n\t total reward: {},\n\t mean reward: {},\n\t sim time: {}".format(e, self.num_episodes, t, np.sum(rewards), np.mean(rewards), time.time() - walltime))
            self.lg.print(episode_str.format(e, self.num_episodes, t, np.sum(rewards),
                  np.mean(rewards), np.max(rewards), np.min(rewards),
                    self._agent.epsilon, time.time() - walltime), lvl="info")

        # self.lg.print("---------------------------")
        # self.lg.print("||  Inference Completed  ||")
        self.lg.print("|------------| INFERENCE COMPLETE |------------|", lvl="info")


    def run_demonstrations(self, train_mode=False, load=True):
        # if self._args['system']['os'] == 'linux':
        #     self.lg.print("-- Can't run demonstration collection on Linux Headless --")

        #TODO make arg
        max_trajectories = 100

        states_taken = []
        observations_taken = []
        actions_taken = []

        if load:
            try:
                states_taken = np.load("demonstrated_states.npz")
                observations_taken = np.load("demonstrated_observations.npz")
                actions_taken = np.load("demonstrated_actions.npz")
            except Exception as e:
                self.lg.print("Could not load npz from file: " + str(e))
        
        for i in range(max_trajectories):
            pause = input("Press enter to start demonstration collection")
            done = False
            self._env.reset(train_mode=train_mode)
            episode_states = []
            episode_observations = []
            episode_actions = []
            while not done:
                brainInf = self._env.step()['DroneBrain']
                ob = brainInf.visual_observations[0]
                state = brainInf.vector_observations
                action = brainInf.previous_vector_actions[0]
                done = brainInf.local_done[0]
                # norm = np.linalg.norm(states[0][3:6] - states[0][9:12])
                # done = norm < threshold
                episode_states.append(state)
                episode_observations.append(ob)
                episode_actions.append(action)
            save  = input('Save this trajectory(y/n): ') == 'y'
            if save:
                states_taken.append(episode_states)
                observations_taken.append(episode_observations)
                actions_taken.append(episode_actions)
            end = input('Stop collecting Data(y/n): ') == 'y'
            if end:
                break
        # states_taken.extend(np.ndarray.tolist(states))
        # actions_taken.extend(np.ndarray.tolist(actions))
        self.lg.print("-- Saving data --")
        states_taken = np.array(states_taken)
        observations_taken = np.array(observations_taken)
        actions_taken = np.array(actions_taken)
        np.savez_compressed("demonstrated_states", states_taken)
        np.savez_compressed("demonstrated_observations", observations_taken)
        np.savez_compressed("demonstrated_actions", actions_taken)

        self.lg.print("-- Data successfully saved --")




