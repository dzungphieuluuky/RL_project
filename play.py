from agents.simple_bandit import SimpleBandit
import argparse

def main():
    parser = argparse.ArgumentParser(description="Play a simple bandit game.")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to play")
    parser.add_argument("--ucb", action='store_true', help="Use Upper Confidence Bound (UCB) for action selection")
    args = parser.parse_args()

    # Initialize the bandit environment
    env = None  # Replace with actual environment initialization if needed
    agent = SimpleBandit(env, actions=10, epsilon=0.1)

    # Play the game
    agent.play(num_episodes=args.num_episodes, ucb=args.ucb)

if __name__ == "__main__":
    main()