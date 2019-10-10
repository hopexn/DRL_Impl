import gym
import numpy as np


class NormalizedWrapper(gym.ActionWrapper):
    def action(self, action):
        high = self.action_space.high
        low = self.action_space.low
        new_action = low + 0.5 * (high - low) * (action + 1)
        new_action = np.clip(new_action, low, high)
        return new_action
    
    def reverse_action(self, action):
        high = self.action_space.high
        low = self.action_space.low
        origin_action = -1 + 2 * (action - low) / (high - low)
        origin_action = np.clip(origin_action, low, high)
        return origin_action
