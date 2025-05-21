from .base_agent import BaseAgent
from collections import defaultdict, Counter
from utils.getter import get_best_action, get_best_value

import random
import numpy as np

class DPAgent(BaseAgent):
    def __init__(self, env, gamma = 0.99, epsilon = 0.7):
        super().__init__()
        self.env = env
        self.action_space = env.action_space.n
        self.observation_space = env.observation_space.n

        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = defaultdict(float)
        # q_table[(state, action)] = value
        self.reward_table = defaultdict(float)
        # reward_table[(state, action, next_state)] = reward
        self.transition_table = defaultdict(Counter)
        # transition_table[(state, action)] = {next_state: number of times get here}

    def select_action(self, state):
        if random.randint(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            best_action = get_best_action(self.q_table, state)
            return best_action

    def update(self):
        for state in range(self.observation_space):
            for action in range(self.action_space):
                action_val = 0
                next_states_counts = self.transition_table[(state, action)]
                total_count = sum(next_states_counts.values)
                for next_state, count in next_states_counts.items:
                    reward = self.reward_table[(state, action, next_state)]
                    new_val = reward + self.gamma * get_best_value(self.q_table, next_state)
                    action_val += (count / total_count) * new_val
                self.q_table[(state, action)] = action_val

    def load_table(self, path):
        self.q_table = np.load(path)
    
    def save_table(self, path):
        np.save(path, self.q_table)