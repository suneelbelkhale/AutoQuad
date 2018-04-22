from unityagents import UnityEnvironment
import numpy as np
import time
import sys

class RunAgent:

    def __init__(self, agent, args):

        self._agent = agent
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
        self.train_after_episode = args['train']['train_after_episode']
        self.model_file = args['train']['model_file']
        #inference
        self.only_inference = args['inference']['only_inference']
        self.num_inference_episodes = args['inference']['num_inference_episodes']

    def create_env(self, file_name):
        self._env = UnityEnvironment(file_name=file_name, worker_id=0)

    def run(self):
        if self.only_inference:
            print("-- Running INFERENCE on %d episodes of length %d -- \n" % (self.num_inference_episodes, self.max_episode_length))
            self.run_inference()
        else:
            print("-- Running TRAINING on %d episodes of length %d -- \n" % (self.num_episodes, self.max_episode_length))
            self.run_training()
    
    def run_training(self, train_mode=True): #batch_size=32, num_episodes=1, max_episode_length=1000, train_period=3, train_after_episode=False, train_mode=False):

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



