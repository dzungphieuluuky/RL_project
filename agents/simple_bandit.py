import numpy as np
import random

from .base_agent import BaseAgent

class SimpleBandit(BaseAgent):
    def __init__(self, env, actions, epsilon):
        super().__init__(env)
        self.number_of_actions = actions
        self.epsilon = epsilon
        self.q = np.zeros(self.number_of_actions)
        self.action_choice = np.zeros(self.number_of_actions)
        self.rewards_means = np.random.uniform(low=-1, high=1, size=[self.number_of_actions])
        self.rewards_stds = np.random.uniform(low=0.1, high=1, size=[self.number_of_actions])
    
    def bandit(self, action):
        return np.random.normal(self.rewards_means[action], self.rewards_stds[action])
    
    def select_action(self):
        if random.randint(0, 1) < self.epsilon:
            action = random.randint(0, self.number_of_actions - 1)
        else:
            action = np.argmax(self.q)
        return action
    
    def ucb_select_action(self):
        ucb_values = self.q + np.sqrt(2 * np.log(np.sum(self.action_choice)) / (self.action_choice + 1e-5))
        return np.argmax(ucb_values)
    
    def play(self, num_episodes = None, ucb=False):
        if num_episodes is None:
            while True:
                if ucb:
                    action = self.ucb_select_action()
                else:
                    action = self.select_action()
                reward = self.bandit(action)
                self.action_choice[action] += 1
                self.q[action] = self.q[action] + (reward - self.q[action])/self.action_choice[action]
                print(f"Action: {action}, Reward: {reward}, Q-value: {self.q[action]}")
        else:
            for i in range(num_episodes):
                if ucb:
                    action = self.ucb_select_action()
                else:
                    action = self.select_action()
                reward = self.bandit(action)
                self.action_choice[action] += 1
                self.q[action] = self.q[action] + (reward - self.q[action])/self.action_choice[action]
                print(f"Episode {i+1}: Action: {action}, Reward: {reward}, Q-value: {self.q[action]}")
                if i % 100 == 0:
                    print(f"Progress: {i}/{num_episodes}")