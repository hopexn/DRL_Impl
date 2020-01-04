import gym

from agents.dqn.DQNAgent import *

env = gym.make("CartPole-v0")

print("Observation space: {}".format(env.observation_space))
print("Action space: {}".format(env.action_space))

nb_actions = env.action_space.n
observation_shape = env.observation_space.shape

train_policy = DecayEpsGreedyQPolicy(eps_min=0, eps_decay=0.99)

agent = DQNAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    train_policy=train_policy,
    dueling_type='max'
)

print("Start training~")
for episode in range(200):
    episode_rewards = 0
    observation = env.reset()
    for step in range(200):
        action = agent.forward(observation)
        
        next_observation, reward, terminal, _ = env.step(action)
        
        agent.backward(observation, action, reward, terminal, next_observation)
        episode_rewards += reward
        
        observation = next_observation
        if terminal:
            break
    
    print("Episode {}: {}".format(episode, episode_rewards))

agent.switch_mode()
print("Start testing~")
for episode in range(10):
    episode_rewards = 0
    observation = env.reset()
    for step in range(200):
        action = agent.forward(observation)
        
        next_observation, reward, terminal, _ = env.step(action)
        
        episode_rewards += reward
        
        observation = next_observation
        if terminal:
            break
        env.render()
    
    print("Episode {}: {}".format(episode, episode_rewards))

env.close()
