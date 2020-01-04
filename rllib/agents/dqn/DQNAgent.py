import tensorflow as tf
from gym.spaces import Box, Discrete

from core.agent import Agent
from core.memory import MemoryNP
from core.policy import *
from utils.common import polyak_averaging


class DQNAgent(Agent):
    
    def __init__(self,
                 observation_space: Box,
                 action_space: Discrete,
                 train_policy=None,
                 test_policy=None,
                 lr=3e-4,
                 gamma=0.99,
                 memory_size=10000,
                 target_model_update=0.99,
                 training=True,
                 enable_double_dqn=True,
                 dueling_type=None):
        
        super().__init__(observation_space, action_space)
        
        # 学习率
        self.lr = lr
        # 衰减系数
        self.gamma = gamma
        # 目标模型更新的频率，若`target_model_update < 1`使用软更新， `target_model_update >= 1`使用硬更新
        self.target_model_update = target_model_update
        # 训练过程使用的策略
        if train_policy is None:
            self.train_policy = DecayEpsGreedyQPolicy()
        else:
            self.train_policy = train_policy
        
        # 测试过程使用的策略
        if test_policy is None:
            self.test_policy = GreedyQPolicy()
        else:
            self.test_policy = test_policy
        
        # 用于标记agent的状态， True表示训练状态， False表示测试状态, 根据状态选择策略
        self.training = training
        self.policy = None
        self.switch_mode(self.training)
        
        # 是否使用double dqn
        self.enable_double_dqn = enable_double_dqn
        # 是否使用dueling dqn
        self.dueling_type = dueling_type
        
        # 动作个数
        self.nb_actions = action_space.n
        # 由于动作是离散的，可以使用一个值来表示
        self.action_shape = (1,)
        # 观测状态的形状
        self.observation_shape = observation_space.shape
        
        # ReplayBuffer
        self.memory = MemoryNP(
            capacity=memory_size,
            action_shape=self.action_shape,
            observation_shape=self.observation_shape
        )
        
        # 创建DNN模型，以及目标模型，初始化参数
        self.model, self.target_model = self.build_all_models()
        
        # 计数
        self.step_count = 0
    
    def build_all_models(self):
        model = self.build_q_net()
        target_model = self.build_q_net()
        
        model = self.use_dueling_network(model)
        target_model = self.use_dueling_network(target_model)
        
        target_model.set_weights(model.get_weights())

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr),
            metrics=['mse'],
            loss=tf.keras.losses.mean_squared_error
        )
        
        return model, target_model
    
    def use_dueling_network(self, model):
        layer = model.layers[-2]
        y = tf.keras.layers.Dense(self.nb_actions + 1, activation='linear')(layer.output)
        if self.dueling_type == 'avg':
            output_layer = tf.keras.layers.Lambda(
                lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.reduce_mean(a[:, 1:], axis=1, keepdims=True),
                output_shape=(self.nb_actions,))(y)
        elif self.dueling_type == 'max':
            output_layer = tf.keras.layers.Lambda(
                lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.math.max(a[:, 1:], axis=1, keepdims=True),
                output_shape=(self.nb_actions,))(y)
        elif self.dueling_type == 'naive':
            output_layer = tf.keras.layers.Lambda(
                lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:],
                output_shape=(self.nb_actions,))(y)
        else:
            output_layer = model.layers[-1].output
        
        model = tf.keras.models.Model(inputs=model.input, outputs=output_layer)
        
        return model
    
    def build_q_net(self):
        '''
        创建深度Q网络，若需要改变网络的容量，可以重载该函数，不过需要保持输入输出的一致
        :return:  q_net
        '''
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=self.observation_shape),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.nb_actions, activation='linear')
        ])
        
        return model
    
    def switch_mode(self, training=None):
        '''
        切换动作策略：
        :param training:  agent所处的模式，
            training=True： 训练模式
            training=False: 测试模式
        '''
        if training is None:
            self.training = ~self.training
        else:
            self.training = training
        
        if self.training:
            self.policy = self.train_policy
            print("Switch to train mode.")
        else:
            self.policy = self.test_policy
            print("Switch to test mode.")
    
    def forward(self, observation):
        '''
        根据观测状态选择动作
        :param observation: 观测状态
        :return:
        '''
        observation = np.expand_dims(observation, axis=0)
        q_values = self.model.predict(observation).squeeze(0)
        action = self.policy.select_action(q_values)
        return action
    
    def backward(self, observation, action, reward, terminal, next_observation):
        '''
        每次与环境交互一次，可以生成一个MDP转移元组样本，保存该元组至ReplayBuffer，采样并训练
        :param observation:
        :param action:
        :param reward:
        :param terminal:
        :param next_observation:
        :return:
        '''
        # 保存样本
        self.memory.store_transition(observation, action, reward, terminal, next_observation)
        
        # 更新Q网络
        if self.enable_double_dqn:
            self.update_model_double_dqn()
        else:
            self.update_model()
        
        # 更新目标Q网络
        self.update_target_model()
    
    def update_model(self):
        # 从ReplayBuffer采样
        observations, actions, rewards, terminals, next_observations = self.memory.sample_batch()
        
        # 计算目标Q值
        target_q_values = np.max(self.target_model.predict(next_observations), axis=1, keepdims=True)
        actions = tf.keras.utils.to_categorical(actions, num_classes=self.nb_actions).astype(np.bool)
        q_values = self.model.predict(observations)
        q_values[actions, np.newaxis] = rewards + self.gamma * target_q_values * (~terminals)
        
        # 更新Q网络
        self.model.fit(observations, q_values, verbose=0)
    
    def update_model_double_dqn(self):
        # 从ReplayBuffer采样
        observations, actions, rewards, terminals, next_observations = self.memory.sample_batch()
        
        # 计算目标Q值
        q_values = self.model.predict(observations)
        q_values_next = self.model.predict(next_observations)
        
        target_action = tf.keras.utils.to_categorical(
            np.argmax(q_values_next, axis=1),
            num_classes=self.nb_actions
        ).astype(np.bool)
        
        target_q_values = self.target_model.predict(next_observations)[target_action].reshape(-1, 1)
        
        actions = tf.keras.utils.to_categorical(actions, num_classes=self.nb_actions).astype(np.bool)
        q_values[actions, np.newaxis] = rewards + self.gamma * target_q_values * (~terminals)
        
        # 更新Q网络
        self.model.fit(observations, q_values, verbose=0)
    
    def update_target_model(self):
        if self.target_model_update < 1.:
            # soft update： w'(t+1) = w'(t) * lamda + w(t) * (1 - lamda)
            new_target_model_weights = polyak_averaging(
                weights_list=self.model.get_weights(),
                target_weights_list=self.target_model.get_weights(),
                polyak=self.target_model_update
            )
            self.target_model.set_weights(new_target_model_weights)
        else:
            # hard update: w'(t+1) = w(t)
            self.step_count += 1
            if self.step_count % int(self.target_model_update) == 0:
                self.target_model.set_weights(self.model.get_weights())
