import gymnasium as gym
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from agents.q_learning import QLearningAgent

def train_blackjack(episodes = 10000):
    agent = QLearningAgent(
        env=gym.make("Blackjack-v1"),
        alpha=0.01,
        gamma=0.99,
        epsilon=1,
        epsilon_decay=0.999
    )
    agent.train(episodes, threshold=1e-8, log_interval=100)
    
    os.makedirs("../models", exist_ok=True)
    agent.save_table("../models/blackjack_q_table.npy")
    print("Training completed.")
    return agent

def play_blackjack(episodes=10, render=True):
    render_mode = "human" if render else None
    agent = QLearningAgent(
        env=gym.make("Blackjack-v1", render_mode=render_mode),
        alpha=0.01,
        gamma=0.99,
        epsilon=1,
        epsilon_decay=0.999
    )
    agent.load_table("models/blackjack_q_table.npy")
    total_reward = 0
    wins = 0

    for _ in range(episodes):
        state, _ = agent.env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, trunc, info = agent.env.step(action)

            if done or trunc:
                break
            episode_reward += reward
            state = next_state
        
        total_reward += episode_reward
        if episode_reward > 0:
            wins += 1
            print("Win Game!")
    print("Done playing!")    

def main():
    parser = argparse.ArgumentParser(description="Blackjack Q-learning")
    parser.add_argument("mode", choices=["train", "play"], help="Mode: train or play")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes playing")
    
    args = parser.parse_args()

    if args.mode == "train":
        episodes = args.episodes or 10000
        train_blackjack(episodes=episodes)
    elif args.mode == "play":
        episodes = args.episodes or 10
        play_blackjack(episodes=episodes)

if __name__ == "__main__":
    main()