from unityagents import UnityEnvironment
import numpy as np

class RunAgent:

    def __init__(self, agent, env_name):
        self._agent = agent
        self.create_env(env_name)

    def create_env(self, file_name):
        self._env = UnityEnvironment(file_name=file_name, worker_id=0)
    
    def run(self, batch_size=32, num_episodes=1, max_episode_length=1000, train_period=3, train_after_episode=False, train_mode=False):

        for e in range(num_episodes):
            #reset
            brainInf = self._env.reset(train_mode=train_mode)['DroneBrain']
            # import ipdb; ipdb.set_trace()

            p_observation = self._agent.preprocess_observation(brainInf.visual_observations[0])

            rewards = []
            done = False
            
            print("-- Episode %d --" % e)

            for time in range(max_episode_length):
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
                if not train_after_episode and len(self._agent.replay_buffer) > batch_size:
                    if time % train_period == 0:
                        self._agent.train(batch_size)

                if done:
                    print("episode: {}/{}, step: {}, reward: {}".format(e, num_episodes, time, np.mean(rewards)))
                    break


                p_observation = next_p_observation
                brainInf = nextBrainInf

            # train after episode
            if train_after_episode and len(self._agent.replay_buffer) > batch_size:
                for i in range(time // train_period):
                    self._agent.train(batch_size)
