from unityagents import UnityEnvironment
import numpy as np
import time
import sys

from unityagents import BrainInfo

class RunAgent:

    def __init__(self, agent, args, demonstrations=False):

        self._agent = agent
        if demonstrations:
            self.env_name = args['system']['demonstration_drone_sim']
        else:
            self.env_name = args['system']['drone_sim']
        self._args = args
        self.create_env(self.env_name)
        self.parseArgs(self._args)

    def parseArgs(self, args):
        #training
        self.max_episode_length = args['train']['max_episode_length']
        self.num_episodes = args['train']['num_episodes']
        self.batch_size = args['train']['batch_size']
        self.train_period = args['train']['train_period']
        self.train_on_demonstrations = args['train']['train_on_demonstrations']
        self.train_after_episode = args['train']['train_after_episode']
        self.model_file = args['train']['model_file']
        #inference
        self.only_inference = args['inference']['only_inference']
        self.num_inference_episodes = args['inference']['num_inference_episodes']
        #demonstrations

    def create_env(self, file_name):
        self._env = UnityEnvironment(file_name=file_name, worker_id=0)

    #gives them pretty high reward
    def train_demonstrations(self):
        print("Beginning Training on demonstrations")

        states_taken = np.load("demonstrated_states.npz")
        observations_taken = np.load("demonstrated_observations.npz")
        actions_taken = np.load("demonstrated_actions.npz")
        
        total_seen = 0

        for ep in range(len(states_taken)):
            these_states = states_taken[ep][0]
            these_observations = observations_taken[ep][0]
            these_actions = actions_taken[ep][0]

            print("Training on demonstration:", ep)
            #loop over episode
            for i in range(len(these_states) - 1):
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
                reward += (0.9 ** (len(these_states) - 1 - i)) * 20000

                sample = ((these_states, these_observations),
                            these_actions,
                            reward,
                            (next_states, next_observations),
                            done
                        )

                self._agent.store_sample(sample)
                total_seen += 1

                if total_seen > self.batch_size:
                    if total_seen % self.freqency == 0:
                        self._agent.train(batch_size=self.batch_size)


                these_states = next_states
                these_observations = next_observations
                these_actions = next_actions


    def run(self, load=False):
        if self.only_inference:
            print("-- Running INFERENCE on %d episodes of length %d -- \n" % (self.num_inference_episodes, self.max_episode_length))
            self.run_inference()
        else:
            print("-- Running TRAINING on %d episodes of length %d -- \n" % (self.num_episodes, self.max_episode_length))
            self.run_training(load)
    
    def run_training(self, train_mode=True, load=False): #batch_size=32, num_episodes=1, max_episode_length=1000, train_period=3, train_after_episode=False, train_mode=False):

        if load:
            try:
                self._agent.load(self.model_file)
            except Exception as e:
                print("Could not load from file:", str(e))

        if self.train_on_demonstrations:
            self.train_demonstrations()

        for e in range(self.num_episodes):
            walltime = time.time()

            #reset
            brainInf = self._env.reset(train_mode=train_mode)['DroneBrain']
            # import ipdb; ipdb.set_trace()

            p_observation = self._agent.preprocess_observation(brainInf.visual_observations[0])

            rewards = []
            done = False
            
            print("-- Episode %d --" % e)
            sys.stdout.flush()

            for t in range(self.max_episode_length):
                #generalized act function takes in state and observations (images)

                action = self._agent.act(brainInf.vector_observations, p_observation)

                nextBrainInf = self._env.step(action)['DroneBrain']
                
                done = brainInf.local_done[0]
                #print(brainInf.local_done)
                reward = self._agent.compute_reward(brainInf, nextBrainInf, action)
                rewards.append(reward)

                next_p_observation = self._agent.preprocess_observation(nextBrainInf.visual_observations[0])

                #stores processed things
                self._agent.store_sample((  (brainInf.vector_observations[0], p_observation), 
                                            action,
                                            reward,
                                            (nextBrainInf.vector_observations[0], next_p_observation),
                                            done    ))


                #train every experience here
                if not self.train_after_episode and len(self._agent.replay_buffer) > self.batch_size:
                    if t % self.train_period == 0:
                        self._agent.train(self.batch_size)

                if t % 100 == 0:
                    #print("step", t)
                    sys.stdout.flush()

                if done:
                    print("episode terminated: {}/{}, step: {}, mean reward: {}, time: {}".format(e, self.num_episodes, t, np.mean(rewards), time.time() - walltime))
                    sys.stdout.flush()
                    break


                p_observation = next_p_observation
                brainInf = nextBrainInf

            # train after episode
            if self.train_after_episode and len(self._agent.replay_buffer) > self.batch_size:
                walltime = time.time()
                for i in range(t // self.train_period):
                    self._agent.train(self.batch_size)
                print("training complete in: {}".format(time.time() - walltime))
                sys.stdout.flush()

            # save after an episode
            if e % 10 == 0:
                self._agent.save(self.model_file)

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
            
            print("-- Episode %d --" % e)

            for t in range(self.max_episode_length):
                #generalized act function takes in state and observations (images)

                action = self._agent.act(brainInf.vector_observations, p_observation)

                nextBrainInf = self._env.step(action)['DroneBrain']
                
                done = brainInf.local_done[0]
                # print(brainInf.local_done)
                reward = self._agent.compute_reward(brainInf, nextBrainInf, action)
                rewards.append(reward)

                if done:
                    print("episode terminated: {}/{}, step: {}, reward: {}, time: {}".format(
                        e, self.num_episodes, t, np.mean(rewards), time.time() - walltime))
                    break

                brainInf = nextBrainInf

        print("---------------------------")
        print("||  Inference Completed  ||")
        print("---------------------------")


    def run_demonstrations(self, train_mode=False, load=True):
        # if self._args['system']['os'] == 'linux':
        #     print("-- Can't run demonstration collection on Linux Headless --")

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
                print("Could not load npz from file: " + str(e))
        
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
        print("-- Saving data --")
        states_taken = np.array(states_taken)
        observations_taken = np.array(observations_taken)
        actions_taken = np.array(actions_taken)
        np.savez_compressed("demonstrated_states", states_taken)
        np.savez_compressed("demonstrated_observations", observations_taken)
        np.savez_compressed("demonstrated_actions", actions_taken)

        print("-- Data successfully saved --")




