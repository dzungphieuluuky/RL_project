import gymnasium as gym
import json
import os
import argparse

from ..agents import QLearningAgent
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def train_q_learning(env_name, config_path=None, save_path=None):
    """Train Q-Learning agent on specified environment"""
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = {
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "num_episodes": 10000
        }
    
    # Create environment
    env = gym.make(env_name)
    
    # Initialize agent
    agent = QLearningAgent(
        env=env,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"]
    )
    
    # Setup logging
    logger = SummaryWriter(comment=f"Q_Learning_{env_name}")
    
    print(f"Training Q-Learning on {env_name}")
    print(f"Config: {config}")
    
    # Train agent
    results = agent.train(
        num_episodes=config["num_episodes"],
        log_interval=100
    )
    
    # Save trained model
    if save_path:
        agent.save_table(save_path)
        logger.info(f"Model saved to {save_path}")
    
    # Print results
    print(f"Training completed!")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Average reward (last 100 episodes): {results['avg_reward']:.4f}")
    
    return agent, results

def main():
    parser = argparse.ArgumentParser(description="Train Q-Learning agent")
    parser.add_argument("--env", default="FrozenLake-v1", help="Environment name")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--save", help="Path to save trained model")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    save_path = args.save or f"results/q_learning_{args.env.replace('-', '_')}.npy"
    
    agent, results = train_q_learning(
        env_name=args.env,
        config_path=args.config,
        save_path=save_path
    )

if __name__ == "__main__":
    main()