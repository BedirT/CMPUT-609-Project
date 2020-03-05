import numpy as np

class agent_REINFORCE:
    def __init__(self, feature_space, action_space, alpha = 0.0001, gamma = 0.99):
        self.alpha = alpha
        self.gamma = gamma
    
        self.feature_space = feature_space
        self.action_space = action_space

        self._reset_weights()

    def step(self, obs):
        probs = self._policy(obs)[0]
        action = np.random.choice(self.action_space, p = probs)
        return action, probs

    def _policy(self, obs):
        z = obs.T.dot(self.theta)
        exp = np.exp(z)
        return exp/np.sum(exp)

    # Jacobian softmax
    def _softmax_grad(self, softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def grad(self, probs, obs, action):
        dsoftmax = self._softmax_grad(probs)[action]
        dlog = dsoftmax / probs[action]
        grad = obs.dot(dlog.reshape([self.action_space, dlog.shape[0]]))
        return grad

    def update(self, grads, rewards):
        for i in range(len(grads)):
            self.theta += self.alpha * grads[i] * \
                sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])

    def _reset_weights(self):
        self.theta = np.random.rand(self.action_space, self.feature_space)