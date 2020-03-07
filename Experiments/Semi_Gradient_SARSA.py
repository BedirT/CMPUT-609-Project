import numpy as np
import random

class SG_SARSA:
    def __init__(self, feature_space, action_space, alpha = 0.0001, gamma = 0.99, eps = .1):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
        self.feature_space = feature_space
        self.action_space = action_space

        self.reset_weights()

    def step(self, obs):
        if np.random.sample() > self.eps:
            return np.argmax(self._act(obs))
        else:
            return np.random.randint(0, self.action_space)

    def _act(self, obs):
        q_vals = np.zeros(self.action_space)
        for a in range(self.action_space):
            q_vals[a] = self.q_hat(obs, a)
        return q_vals

    def q_hat(self, obs, action):
        q_val = self.w.T.dot(self._x(obs, action))
        return q_val

    def grad_q_hat(self, obs, action):
        return self._x(obs, action)

    def _x(self, obs, action):
        one_hot = np.zeros_like(self.w)
        j = 0
        for i in range(action * self.feature_space, ((action+1) * self.feature_space)):
            one_hot[i] = obs[j]
            j += 1
        return one_hot

    def update(self, observations, actions, rewards):
        for i in range(len(observations)):
            G = sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])
            self.w += self.alpha * self.grad_q_hat(observations[i], actions[i]) * (G - self.q_hat(observations[i], actions[i]))

    def reset_weights(self):
        # self.w = np.random.rand(self.feature_space, self.action_space) 
        self.w = np.zeros(self.feature_space * self.action_space)