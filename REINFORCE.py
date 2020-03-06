import numpy as np

class agent_REINFORCE:
    def __init__(self, feature_space, action_space, alpha = 0.0001, gamma = 0.99):
        self.alpha = alpha
        self.gamma = gamma
    
        self.feature_space = feature_space
        self.action_space = action_space

        self.reset_weights()

    def step(self, obs):
        probs = self._policy(obs)
        action = np.random.choice(self.action_space, p = probs)
        return action

    def _policy(self, obs):
        probs = np.zeros(self.action_space)
        for a in range(self.action_space):
            probs[a] = np.exp(self.theta.T.dot(self._x(obs, a)))
        return probs/np.sum(probs)

    def _gradient(self, obs, action):
        grads = []
        probs = self._policy(obs)
        for b in range(self.action_space):
            grads.append(probs[b] * self._x(obs, b))
        return self._x(obs, action) - np.sum(grads)

    def _x(self, obs, action):
        one_hot = np.zeros_like(self.theta)
        j = 0
        for i in range(action * self.feature_space, ((action+1) * self.feature_space)):
            one_hot[i] = obs[j]
            j += 1
        return one_hot

    # Jacobian softmax
    # def _softmax_grad(self, softmax):
    #     s = softmax.reshape(-1,1)
    #     return np.diagflat(s) - np.dot(s, s.T)

    # def grad(self, probs, obs, action):
    #     dsoftmax = self._softmax_grad(probs)[action, :]
    #     print('dsoftmax---', dsoftmax.shape)
    #     dlog = dsoftmax / probs[action]
    #     print('dlog---', dlog.shape)
    #     grad = obs.T.dot(dlog[None,:])
    #     return grad

    def update(self, observations, actions, rewards):
        for i in range(len(observations)):
            self.theta += self.alpha * self._gradient(observations[i], actions[i]) * \
                sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])

    def reset_weights(self):
        self.theta = np.random.rand(self.feature_space * self.action_space)