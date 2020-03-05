import numpy as np

class agent_REINFORCE:
    def __init__(self, feature_space, action_space, alpha = 0.0001, gamma = 0.99):
        self.alpha = alpha
        self.gamma = gamma
    
        self.feature_space = feature_space
        self.action_space = action_space

        self.theta = np.random.rand(self.feature_space, 1)

    def step(self, state):
        probs = self._policy(state)
        action = np.random.choice(self.action_space, p = probs)
        return action, probs

    def _policy(self, state):
        probs = []
        for a in range(self.action_space):
            probs.append(self.theta.T.dot(self._x(state, a))[0][0])
        probs = np.exp(probs)
        return probs/np.sum(probs)
    
    def _x(self, state, action):
        one_hot = state
        one_hot[self.feature_space - self.action_space + action] = 1
        return one_hot
        # return state

    # Jacobian softmax
    def _softmax_grad(self, softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def grad(self, probs, state, action):
        dsoftmax = self._softmax_grad(probs)[action]
        dlog = dsoftmax / probs[action]
        grad = state.dot(dlog.reshape([1, dlog.shape[0]]))
        return grad

    def update(self, grads, rewards):
        print(grads[0].shape)
        for i in range(len(grads)):
            self.theta += self.alpha * grads[i] * \
                sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])