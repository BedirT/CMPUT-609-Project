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
            probs[a] = self.theta.T.dot(self._x(obs, a))
        probs = np.exp(probs)
        # print(probs)
        return probs/np.sum(probs)

    def _gradient(self, obs, action):
        grads = np.zeros_like(self.theta)
        probs = self._policy(obs)
        for b in range(self.action_space):
            grads += self._x(obs, b) * probs[b]
        return self._x(obs, action) - grads

    def _x(self, obs, action):
        one_hot = np.zeros_like(self.theta)
        j = 0
        for i in range(action * self.feature_space, ((action+1) * self.feature_space)):
            one_hot[i] = obs[j]
            j += 1
        # print(one_hot)
        return one_hot

    def update(self, observations, actions, rewards):
        for i in range(len(observations)):
            self.theta += self.alpha * self._gradient(observations[i], actions[i]) * \
                sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])

 
    def reset_weights(self):
        self.theta = np.random.rand(self.feature_space * self.action_space)
        
        