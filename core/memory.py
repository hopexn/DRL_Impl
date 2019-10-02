import random
from collections import deque

import numpy as np


class Memory:
    def __init__(self, capacity, observation_shape, action_shape):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        self.memory = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.memory)
    
    def store_transition(self, observation, action, reward, terminal, next_observation):
        self.memory.append((observation, action, reward, terminal, next_observation))
    
    def sample_batch(self, batch_size=32):
        batch_size = min(batch_size, len(self.memory))
        batch_samples = random.sample(self.memory, k=batch_size)
        
        observations, actions, rewards, terminals, next_observations = [], [], [], [], []
        for sample in batch_samples:
            observations.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            terminals.append(sample[3])
            next_observations.append(sample[4])
        
        observations = np.array(observations).reshape((batch_size,) + self.observation_shape)
        actions = np.array(actions).reshape((batch_size,) + self.action_shape)
        rewards = np.array(rewards).reshape((batch_size, 1))
        terminals = np.array(terminals).reshape((batch_size, 1))
        next_observations = np.array(next_observations).reshape((batch_size,) + self.observation_shape)
        
        return observations, actions, rewards, terminals, next_observations
