# This code creates the Lunar Lander environment and trains the agent using the DQN algorithm.

import gymnasium as gym

from Wrappers import GymnasiumWrapper
class LunarLanderWrapper(GymnasiumWrapper):
    def __init__(self, env):
        super(LunarLanderWrapper, self).__init__(env)
        self.env = env
        self.state = None
    
    def modify_observation(self, observation):
        return observation
    
    def modify_reward(self, reward):
        return reward
    
    def modify_action(self, action):
        return action


env = gym.make('LunarLander-v2')
wrapped_env = LunarLanderWrapper(env)


from Agent import BasicAgent

parameters = {
    # Q-Network parameters
    "learning_rate": 0.001,
    "batch_size": 64,
    "gamma": 0.99,

    # Epsilon-greedy policy
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,

    # Target network update frequency
    "update_target_every": 1000,

    # PER parameters
    "alpha": 0.6,
    "beta": 0.4,
    "beta_increment_per_sampling": 0.001,
    "capacity": 10000
}

agent = BasicAgent(wrapped_env, parameters)

number_of_episodes = 1000
agent.train(number_of_episodes)
agent.plot_results()