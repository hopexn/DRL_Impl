import tensorflow as tf

from core.agent import Agent
from core.memory import Memory
from core.policies import *


class DQNAgent(Agent):
    
    def __init__(self,
                 nb_actions,
                 observation_shape,
                 gamma=0.99,
                 target_model_update=100):
        super().__init__()
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.observation_shape = observation_shape
        self.target_model_update = target_model_update
        
        self.memory = Memory()
        
        self.model = self._build_network(nb_actions, observation_shape)
        self.target_model = self._build_network(nb_actions, observation_shape)
        self.target_model.set_weights(self.model.get_weights())
        
        self.update_count = 0
    
    def _build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=self.observation_shape),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.nb_actions, activation='linear')
        ])
        model.compile(optimizer='adam', metrics=['mse'], loss=tf.keras.losses.mean_squared_error)
        return model
    
    def forward(self, observation):
        observation = np.expand_dims(observation, axis=0)
        q_values = self.model.predict(observation)
        return q_values
    
    def backward(self, observation, action, reward, terminal, next_observation):
        self.memory.store_transition(observation, action, reward, terminal, next_observation)
        
        observations, actions, rewards, terminals, next_observations = self.memory.sample_batch()
        
        actions = tf.keras.utils.to_categorical(actions, num_classes=self.nb_actions).astype(np.bool)
        
        q_values = self.model.predict(observations)
        
        target_q_values = np.max(self.target_model.predict(next_observations), axis=1)
        q_values[actions] = rewards + self.gamma * target_q_values * (~terminals)
        
        self.model.fit(observations, q_values, verbose=0)
        
        self.update_count += 1
        if self.update_count % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
