import cv2
import numpy as np
import gymnasium as gym


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[34:194, :]
    # N.B. IF INPUT SHAPE IS CHANGED THIS WON'T WORK
    resized = cv2.resize(cropped, (80, 80), interpolation=cv2.INTER_AREA)
    normalised = resized / 255.0
    new_frame = np.expand_dims(normalised, axis=0)
    return new_frame

class AtariWrapper(gym.Wrapper):
    def __init__(self, env, crop_area=(0, -1, 0, -1), resize_shape=(80, 80)):
        super(AtariWrapper, self).__init__(env)
        self.crop_area = crop_area
        self.resize_shape = resize_shape
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        modified_obs = self.modify_observation(obs)
        modified_reward = self.modify_reward(reward)
        return modified_obs, modified_reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        modified_obs = self.modify_observation(obs)
        return modified_obs, info
    
    def modify_action(self, action):
        # Overide in subclasses
        return action
    
    def modify_observation(self, observation):
        gray_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        top, bottom, left, right = self.crop_area
        cropped_observation = gray_observation[top:bottom, left:right]
        resized_observation = cv2.resize(cropped_observation, self.resize_shape, interpolation=cv2.INTER_AREA)
        normalised_observation = resized_observation / 255.0
        observation_new = np.expand_dims(normalised_observation, axis=0)
        return observation_new
    
    def modify_reward(self, reward):
        #Override
        return reward
    
class MsPacmanReducedActionSpaceWrapper(AtariWrapper):
    def __init__(self, env):
        crop_area = (34, 194, 0, -1)
        resize_shape = (80, 80)
        super(MsPacmanReducedActionSpaceWrapper, self).__init__(env, crop_area, resize_shape)
    
    def modify_action(self, action):
        # 0 : No operation
        # 1 : Up
        # 2 : Right
        # 3 : Left
        # 4 : Down
        # -> if in first stage of learning
        # 5 : Up-Right -> 2
        # 6 : Up-Left -> 3
        # 7 : Down-Right -> 2
        # 8 : Down-Left -> 3

        if action in [3, 6, 8]:
            # Move left
            action = 3
        if action in [2, 5, 7]:
            # Move right
            action = 2
        return action
    
class MsPacmanFullActionSpaceWrapper(AtariWrapper):
    def __init__(self, env):
        crop_area = (34, 194, 0, -1)
        resize_shape = (80, 80)
        super(MsPacmanFullActionSpaceWrapper, self).__init__(env, crop_area, resize_shape)

    
class PacmanWrapper(AtariWrapper):
    def __init__(self, env):
        crop_area = (34, 194, 0, -1)
        resize_shape = (80, 80)
        super(PacmanWrapper, self).__init__(env, crop_area, resize_shape)
        self.repeat_count = 4
    

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat_count=4):
        super(ActionRepeatWrapper, self).__init__(env)
        self.repeat_count = repeat_count

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self.repeat_count):
            next_state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                print('DONE)')
                break
        return next_state, total_reward, terminated, truncated, info
    
class PongWrapper(AtariWrapper):
    def __init__(self, env):
        crop_area = (34, 194, 0, -1)  # Crop area specific to Pong
        resize_shape = (80, 80)       # Resize to 80x80
        super(PongWrapper, self).__init__(env, crop_area, resize_shape)

    def modify_action(self, action):
        # Pong's action space usually includes 6 actions:
        # 0: No operation, 1: Fire (same as no-op in Pong), 2: Right, 3: Left, 4: Right Fire, 5: Left Fire
        if action in [0, 5]:
          # Move left
          action = 3
        if action in [1, 4]:
          # Move right
          action = 2
        return action

    def modify_reward(self, reward):
        # UPDATE : encourage longer play as it means survival
        survival_bonus = 0.0001
        return reward + survival_bonus
    
class AlienVeryReducedActionSpaceWrapper(AtariWrapper):
    def __init__(self, env):
        crop_area = (34, 194, 0, -1)
        resize_shape = (80, 80)
        super(AlienVeryReducedActionSpaceWrapper, self).__init__(env, crop_area, resize_shape)

    def modify_action(self, action):
        # 0 : NOOP
        # 1 : FIRE
        # 2 : UP
        # 3 : RIGHT
        # 4 : LEFT
        # 5 : DOWN
        # 6 : UPRIGHT -> 3
        # 7 : UPLEFT -> 4
        # 8 : DOWNRIGHT -> 3
        # 9 : DOWNLEFT -> 4
        # 10 : UPFIRE
        # 11 : RIGHTFIRE
        # 12 : LEFTFIRE
        # 13 : DOWNFIRE
        # 14 : UPRIGHTFIRE
        # 15 : UPLEFTFIRE
        # 16 : DOWNRIGHTFIRE
        # 17 : DOWNLEFTFIRE

        # Movement : UP, RIGHT, LEFT, DOWN
        # Fire : FIRE

        if action in [3, 6, 8]:
            # Move left
            action = 3
        if action in [4, 7, 9]:
            # Move right
            action = 4
        if action in [1, 10, 11, 12, 13, 14, 15, 16, 17]:
            # Fire
            action = 1
        return action

    
class AlienReducedActionSpaceWrapper(AtariWrapper):
    def __init__(self, env):
        crop_area = (34, 194, 0, -1)
        resize_shape = (80, 80)
        super(AlienReducedActionSpaceWrapper, self).__init__(env, crop_area, resize_shape)

    def modify_action(self, action):
        # 0 : NOOP
        # 1 : FIRE
        # 2 : UP
        # 3 : RIGHT
        # 4 : LEFT
        # 5 : DOWN
        # 6 : UPRIGHT
        # 7 : UPLEFT
        # 8 : DOWNRIGHT
        # 9 : DOWNLEFT
        # 10 : UPFIRE
        # 11 : RIGHTFIRE
        # 12 : LEFTFIRE
        # 13 : DOWNFIRE
        # 14 : UPRIGHTFIRE -> 11
        # 15 : UPLEFTFIRE -> 12
        # 16 : DOWNRIGHTFIRE -> 11
        # 17 : DOWNLEFTFIRE -> 12

        # Movement : UP, RIGHT, LEFT, DOWN
        # Fire : FIRE, UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE
        if action in [3, 6, 8]:
            # Move left
            action = 3
        if action in [4, 7, 9]:
            # Move right
            action = 4
        if action in [11, 14, 16]:
            # Fire right
            action = 11
        if action in [12, 15, 17]:
            # Fire left
            action = 12
        return action
    
class AlienSemiReducedActionSpaceWrapper(AtariWrapper):
    def __init__(self, env):
        crop_area = (34, 194, 0, -1)
        resize_shape = (80, 80)
        super(AlienSemiReducedActionSpaceWrapper, self).__init__(env, crop_area, resize_shape)

    def modify_action(self, action):
        # 0 : NOOP
        # 1 : FIRE
        # 2 : UP
        # 3 : RIGHT
        # 4 : LEFT
        # 5 : DOWN
        # 6 : UPRIGHT
        # 7 : UPLEFT
        # 8 : DOWNRIGHT
        # 9 : DOWNLEFT
        # 10 : UPFIRE
        # 11 : RIGHTFIRE
        # 12 : LEFTFIRE
        # 13 : DOWNFIRE
        # 14 : UPRIGHTFIRE -> 11
        # 15 : UPLEFTFIRE -> 12
        # 16 : DOWNRIGHTFIRE -> 11
        # 17 : DOWNLEFTFIRE -> 12

        # Movement : UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT
        # Fire : FIRE, UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE
        if action in [11, 14, 16]:
            # Fire right
            action = 11
        if action in [12, 15, 17]:
            # Fire left
            action = 12
        return action


class AlienFullActionSpaceWrapper(AtariWrapper):
    def __init__(self, env):
        crop_area = (34, 194, 0, -1)
        resize_shape = (80, 80)
        super(AlienFullActionSpaceWrapper, self).__init__(env, crop_area, resize_shape)