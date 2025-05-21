from base_agent import BaseAgent
from collections import defaultdict
from utils.getter import get_best_action
import random
import numpy as np

class SARSAAgent(BaseAgent):
    def __init__(self, env, alpha = 0.01, gamma = 0.99, epsilon = 1):
        super().__init__(env)
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.random.random((self.action_space, self.observation_space))

    def select_action(self, state):
        if random.randint(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            best_action = get_best_action(self.q_table, state)
            return best_action
    
    def update(self, state, action, reward, next_state, next_action):
        old_val = self.q_table[state][action]
        target_val = self.q_table[next_state][next_action]
        self.q_table[state][action] = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * target_val)

    def load_table(self, path):
        self.q_table = np.load(path)
    
    def save_table(self, path):
        np.save(path, self.q_table)
