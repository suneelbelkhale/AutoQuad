from unityagents import UnityEnvironment
import numpy as np

class RunAgent:

    def __init__(self, agent, args):
        self._agent = agent
        self.env_name = args['system']['drone_sim']
        self.create_env(self.env_name)
        self.max_episode_length = args['train']['max_episode_length']
        self.num_episodes = args['train']['num_episodes']
        self.batch_size = args['train']['batch_size']
        self.train_period = args['train']['train_period']
        self.train_after_episode = args['train']['train_after_episode']
        self.model_file = args['train']['model_file']

    def create_env(self, file_name):
        self._env = UnityEnvironment(file_name=file_name, worker_id=0)
    
    def run(self, train_mode=True): #batch_size=32, num_episodes=1, max_episode_length=1000, train_period=3, train_after_episode=False, train_mode=False):

        for e in range(self.num_episodes):
            #reset
            brainInf = self._env.reset(train_mode=train_mode)['DroneBrain']
            # import ipdb; ipdb.set_trace()

            p_observation = self._agent.preprocess_observation(brainInf.visual_observations[0])

            rewards = []
            done = False
            
            print("-- Episode %d --" % e)

            for time in range(self.max_episode_length):
                #generalized act function takes in state and observations (images)

                action = self._agent.act(brainInf.vector_observations[0], p_observation)

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
                    if time % self.train_period == 0:
                        self._agent.train(self.batch_size)

                if done:
                    print("episode: {}/{}, step: {}, reward: {}".format(e, self.num_episodes, time, np.mean(rewards)))
                    break


                p_observation = next_p_observation
                brainInf = nextBrainInf

            # train after episode
            if self.train_after_episode and len(self._agent.replay_buffer) > self.batch_size:
                for i in range(time // self.train_period):
                    self._agent.train(self.batch_size)

            # save after an episode
            self._agent.save(self.model_file)
