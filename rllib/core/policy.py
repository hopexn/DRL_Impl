import numpy as np


class Policy:
    def __init__(self, **kwargs):
        pass
    
    def select_action(self, **kwargs):
        raise NotImplementedError()


class RandomQPolicy(Policy):
    def select_action(self, q_values: np.array):
        assert q_values.ndim == 1
        action = np.random.randint(q_values.shape[0])
        return action


class GreedyQPolicy(Policy):
    def select_action(self, q_values: np.array):
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class EpsGreedyQPolicy(GreedyQPolicy):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.random_policy = RandomQPolicy()
    
    def select_action(self, q_values: np.array):
        assert q_values.ndim == 1
        if np.random.random() < self.eps:
            return self.random_policy.select_action(q_values)
        else:
            return super().select_action(q_values)


class DecayEpsGreedyQPolicy(EpsGreedyQPolicy):
    def __init__(self, eps_decay=0.99, eps_min=0.02):
        super().__init__(1.0)
        self.eps_decay = eps_decay
        self.eps_min = eps_min
    
    def select_action(self, q_values):
        action = super().select_action(q_values)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return action


class BoltzmannQPolicy(Policy):
    def __init__(self, tau=1.):
        super().__init__()
        self.tau = tau
    
    def select_action(self, q_values):
        assert q_values.ndim == 1
        
        exp_values = np.exp((q_values - np.max(q_values)))
        probs = exp_values / np.sum(exp_values)
        
        action = np.random.choice(a=np.arange(len(q_values)), p=probs)
        return action
