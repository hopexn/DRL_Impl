import os

import gym
import numpy as np
import tensorflow as tf
from utils.common import NormalizedWrapper

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.enable_eager_execution()

env = NormalizedWrapper(gym.make("Pendulum-v0"))

print("Observation space: {}".format(env.observation_space.shape))
print("Action space: {}".format(env.action_space.shape))

nb_actions = env.action_space.shape[0]
observation_shape = env.observation_space.shape

# from td3.TD3Agent import TD3Agent
# agent = TD3Agent(env.action_space, env.observation_space, nb_steps_warmup=2000)

# from ddpg.DDPGAgent import DDPGAgent
# agent = DDPGAgent(env.action_space, env.observation_space, nb_steps_warmup=2000)

from agents.sac.SACAgent import SACAgent
agent = SACAgent(env.action_space, env.observation_space, nb_steps_warmup=2000)

print("Start training~")
for episode in range(200):
    episode_rewards = 0
    observation = env.reset()
    observation = observation.reshape(observation_shape).astype(np.double)

    for step in range(200):
        action = agent.forward(observation)

        next_observation, reward, terminal, _ = env.step(action)

        next_observation = next_observation.reshape(observation_shape).astype(
            np.double)
        reward = reward.astype(np.double)

        agent.backward(observation, action, reward, terminal, next_observation)
        episode_rewards += reward

        observation = next_observation

        if terminal:
            break

    print("Episode {}: {}".format(episode, episode_rewards))

agent.training = False
print("Start testing~")
for episode in range(20):
    episode_rewards = 0
    observation = env.reset()
    observation = observation.reshape(observation_shape)

    for step in range(200):
        env.render()
        action = agent.forward(observation)

        next_observation, reward, terminal, _ = env.step(action)
        next_observation = next_observation.reshape(observation_shape)

        episode_rewards += reward
        observation = next_observation

        if terminal:
            break

    print("Episode {}: {}".format(episode, episode_rewards))