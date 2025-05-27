from base_agent import BaseAgent

import numpy as np

class TDAgent(BaseAgent):
    def __init__(self, env, policy, alpha = 0.01, gamma = 0.99):
        super().__init__(env)
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma

        self.action_space = self.env.action_space.n
        self.values = np.random.random(self.action_space)
        
    def select_action(self):
        return np.argmax(self.policy)
    
    def update_per_episode(self):
        done = False
        state = self.env.reset()[0]
        while not done:
            action = self.select_action()
            next_state, reward, done, truncate, info = self.env.step(action)

            if done or truncate:
                break

            old_val = self.values[state]
            target_val = self.values[next_state]
            self.values[state] = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * target_val)
            state = next_state