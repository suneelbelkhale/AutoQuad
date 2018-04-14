from unityagents import UnityEnvironment

class RunAgent:

    def __init__(self, agent, env_name):
        self._agent = agent
        self.create_env(env_name)

    def create_env(self, file_name):
        self._env = UnityEnvironment(file_name=file_name, worker_id=0)
    
    def run(self, batch_size=32, num_episodes=1, max_episode_length=500, train_period=3, train_after_episode=False, train_mode=False):
        done = False

        for e in range(num_episodes):
            #reset
            brainInf = self._env.reset(train_mode=train_mode)['DroneBrain']
            p_observation = self._agent.preprocess_observation(brainInf.observations[0])

            for time in range(max_episode_length):
                #generalized act function takes in state and observations (images)

                action = self._agent.act(brainInf.states[0], p_observation)

                nextBrainInf = self._env.step(action)['DroneBrain']
                
                done = brainInf.local_done
                reward = self._agent.compute_reward(brainInf, nextBrainInf, action)
                next_p_observation = self._agent.preprocess_observation(nextBrainInf.observations[0])

                #stores processed things
                self._agent.store_sample((  (brainInf.states[0], p_observation), 
                                            action,
                                            reward,
                                            (nextBrainInf.states[0], next_p_observation),
                                            done    ))


                #train every experience here
                if not train_after_episode and len(self._agent.replay_buffer) > batch_size:
                    if time % train_period == 0:
                        self._agent.train(batch_size)

                if done:
                    print("episode: {}/{}, score: {}"
                          .format(e, num_episodes, time))
                    break


                p_observation = next_p_observation
                brainInf = nextBrainInf

            # train after episode
            if train_after_episode and len(self._agent.replay_buffer) > batch_size:
                for i in range(time // train_period):
                    self._agent.train(batch_size)
