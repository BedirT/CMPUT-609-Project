import numpy as np
import random

class SG_SARSA:
    def __init__(self, feature_space, action_space, alpha = 0.0001, gamma = 0.99, eps = .2):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
        self.feature_space = feature_space
        self.action_space = action_space

        self._reset_weights()

    def step(self, obs):
        if np.random.sample() > self.eps:
            temp = self._act(state)
            if temp[0] == temp[1] == temp[2] == temp[3]:
                return np.random.randint(0, self.action_space)
            return np.argmax(self._act(obs))
        else:
            return np.random.randint(0, self.action_space)

    def _act(self, obs):
        q_vals = []
        for a in range(self.action_space):
            q_vals.append(self.q_hat(obs, a))
        return q_vals

    def q_hat(self, obs, action):
        q_val = self.w.T.dot(self._x(obs, action))
        return q_val.reshape((q_val.shape[0], 1)) # no need for reshape

    def grad_q_hat(self, obs, action):
        return self._x(obs, action)

    def _x(self, obs, action):
        one_hot = obs
        one_hot[self.feature_space - self.action_space + action] = 1
        return one_hot
        # return obs

    def update(self, observations, actions, rewards):
        for i in range(len(observations)):
            G = sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])
            self.w += self.alpha * (G - self.q_hat(observations[i], actions[i])) * self.grad_q_hat(observations[i], actions[i])

    def _reset_weights(self):
        self.w = np.random.rand(self.feature_space, self.action_space) 