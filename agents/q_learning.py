import random
import numpy as np

from .base_agent import BaseAgent
from torch.utils.tensorboard import SummaryWriter

class QLearningAgent(BaseAgent):
    def __init__(self, env, alpha = 0.01, gamma = 0.99, epsilon = 1, epsilon_decay = 0.9):
        super().__init__(env)
        self.action_space = self.env.action_space.n
        self.observation_space = len(self.env.observation_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((18, 10, 2, self.action_space))

    def get_state_indices(self, state):
        player_sum, dealer_card, usable_ace = state
        return (player_sum - 4, dealer_card - 1, int(usable_ace))
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            indices = self.get_state_indices(state)
            best_action = np.argmax(self.q_table[indices])
            return best_action
    
    def update(self, state, action, reward, next_state):
        indices = self.get_state_indices(state)
        old_val = self.q_table[indices][action]
        
        next_indices = self.get_state_indices(next_state)
        next_q_max = np.max(self.q_table[next_indices])
        target_val = reward + self.gamma * next_q_max
        self.q_table[indices][action] = (1 - self.alpha) * old_val + self.alpha * target_val

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def train(self, num_episodes = None, threshold = 1e-8, log_interval = 100):
        if log_interval is not None:
            writer = SummaryWriter(comment=f"QLearning_{self.env.spec.id}")
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
                
                indices = self.get_state_indices(state)
                old_val = self.q_table[indices][action]
                self.update(state, action, reward, next_state)
                new_val = self.q_table[indices][action]
                
                if different_val is None:
                    different_val = abs(new_val - old_val)
                else: different_val = min(different_val, abs(new_val - old_val))
                state = next_state

            if num_episodes is not None and episode == num_episodes:
                break

            if log_interval is not None and episode % log_interval == 0:
                avg_reward = np.mean([self.q_table[state][action] for state in range(self.observation_space) for action in range(self.action_space)])
                writer.add_scalar('Average Reward', avg_reward, episode)
                print(f'Episode {episode}, Average Reward: {avg_reward}, Epsilon: {self.epsilon:.4f}')

            episode += 1
            if num_episodes is None and different_val <= threshold:
                break
            
    def load_table(self, path):
        self.q_table = np.load(path)
    
    def save_table(self, path):
        np.save(path, self.q_table)
