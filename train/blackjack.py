import gymnasium as gym
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from agents.q_learning import QLearningAgent

def train_blackjack():
    agent = QLearningAgent(
        env=gym.make("Blackjack-v1"),
        alpha=0.01,
        gamma=0.99,
        epsilon=1,
        epsilon_decay=0.999
    )
    agent.train(num_episodes=10000, threshold=1e-8, log_interval=100)
    agent.save_table("blackjack_q_table.npy")
    print("Training completed.")

if __name__ == "__main__":
    train_blackjack()