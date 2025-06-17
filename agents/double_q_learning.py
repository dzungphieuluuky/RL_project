import random
import numpy as np

from .base_agent import BaseAgent

class DoubleQLearningAgent(BaseAgent):
    def __init__(self, env, alpha = 0.01, gamma = 0.99, epsilon = 0.7):
        super().__init__(env)
        self.alpha : float = alpha
        self.gamma : float = gamma
        self.epsilon : float = epsilon

        self.action_space : int = self.env.action_space.n
        self.observation_space : int = self.env.observation_space.n

        self.q1 : np.array = np.random.uniform(low=-1, high=1, size=[self.observation_space, self.action_space])
        self.q2 : np.array = np.random.uniform(low=-1, high=1, size=[self.observation_space, self.action_space])

    def select_action(self, state):
        if random.randint(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return np.argmax(self.q1[state] + self.q2[state])
    
    def update(self):
        state = self.env.reset()[0]
        done = False

        while not done:
            action = self.select_action(state)
            next_state, reward, done, truncate, info = self.env.step(action)

            if random.randint(0, 1) < 0.5:
                self.q1[state][action] = (1 - self.alpha) * self.q1[state][action] + \
                        self.alpha * (reward + self.gamma * self.q2[next_state][np.argmax(self.q1[next_state])])
            else:
                self.q2[state][action] = (1 - self.alpha) * self.q2[state][action] + \
                        self.alpha * (reward + self.gamma * self.q1[next_state][np.argmax(self.q2[next_state])])
            state = next_state
        
