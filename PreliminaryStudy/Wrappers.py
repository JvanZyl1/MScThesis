import gymnasium as gym

class GymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GymnasiumWrapper, self).__init__(env)
        self.env = env
        self.state = None

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        modified_obs = self.modify_observation(obs)
        modified_reward = self.modify_reward(reward)
        return modified_obs, modified_reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        modified_obs = self.modify_observation(obs)
        return modified_obs, info
    
    def modify_observation(self, observation):
        return observation
    
    def modify_reward(self, reward):
        return reward
    
    def modify_action(self, action):
        return action