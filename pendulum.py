import os

import gym
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

env = gym.make("Pendulum-v0")

print("Observation space: {}".format(env.observation_space.shape))
print("Action space: {}".format(env.action_space.shape))

nb_actions = env.action_space.shape[0]
observation_shape = env.observation_space.shape

from td3.TD3Agent import TD3Agent
agent = TD3Agent(env.action_space, env.observation_space, nb_steps_warmup=2000)

# from ddpg.DDPGAgent import DDPGAgent
# agent = DDPGAgent(env.action_space, env.observation_space, nb_steps_warmup=2000)

print("Start training~")
for episode in range(100):
    episode_rewards = 0
    observation = env.reset()
    observation = observation.reshape(observation_shape)
    
    for step in range(200):
        action = agent.forward(observation)
        
        next_observation, reward, terminal, _ = env.step(action)
        next_observation = next_observation.reshape(observation_shape)
        
        agent.backward(observation, action, reward, terminal, next_observation)
        episode_rewards += reward
        
        observation = next_observation
        
        if terminal:
            break
    
    print("Episode {}: {:.3f}".format(episode, episode_rewards))

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
    
    print("Episode {}: {:.3f}".format(episode, episode_rewards))
