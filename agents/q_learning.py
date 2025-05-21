import utils
from base_agent import BaseAgent
from utils.getter import get_best_action

import random
import numpy as np
from collections import defaultdict

class QLearningAgent(BaseAgent):
    def __init__(self, env, alpha = 0.01, gamma = 0.99, epsilon = 1, epsilon_decay = 0.9):
        super().__init__(env)
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(float)

    def select_action(self, state):
        if random.randint(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            best_action = get_best_action(self.q_table, state)
            return best_action
    
    def update(self, state, action, reward, next_state):
        old_val = self.q_table[(state, action)]
        target_val = max(self.q_table[(next_state, )])
        self.q_table[(state, action)] = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * target_val)

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def train(self, threshold = 1e-3):
        episode = 1
        different_val = None
        while True:
            state = self.env.reset()[0]
            done = False
            print(f'Episode: {episode}')
            while not done:
                action = self.select_action(state)
                self.decay_epsilon()
                next_state, reward, done, truncate, info = self.env.step(action)

                if done or truncate:
                    break

                old_val = self.q_table[(state, action)]
                self.update(state, action, reward, next_state)
                new_val = self.q_table[(state, action)]
                
                if different_val is None:
                    different_val = abs(new_val - old_val)
                else: different_val = min(different_val, abs(new_val - old_val))
                state = next_state
            
            if different_val <= threshold:
                break
        print(f'Training successfully!')
            
    def load_table(self, path):
        self.q_table = np.load(path)
    
    def save_table(self, path):
        np.save(path, self.q_table)
