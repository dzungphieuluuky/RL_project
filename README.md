# RL_project

A simple implementation of reinforcement learning algorithms based on "Reinforcement Learning: An Introduction" by Sutton and Barto.

## 🚀 Features

- **Multiple RL Algorithms**: Implementations of Q-Learning, SARSA, Double Q-Learning, Temporal Difference (TD), Dynamic Programming, and Multi-Armed Bandits.
- **Modular Architecture**: Clean separation between agents, utilities, and neural networks.
- **PyTorch Integration**: Neural network support for deep RL algorithms.
- **Extensible Design**: Easy to add new algorithms and environments.
- **Educational Focus**: Well-documented code perfect for learning RL concepts.

## 📁 Project Structure

```
RL_project/
├── agents/                 # RL algorithm implementations
│   ├── base_agent.py      # Abstract base class for all agents
│   ├── q_learning.py      # Q-Learning algorithm
│   ├── sarsa.py           # SARSA algorithm
│   ├── double_q_learning.py # Double Q-Learning
│   ├── TD.py              # Temporal Difference learning
│   ├── dp.py              # Dynamic Programming
│   └── simple_bandit.py   # Multi-Armed Bandit
├── network/               # Neural network components
│   └── simple_network.py  # Basic feedforward network
├── utils/                 # Utility functions
│   └── getter.py          # Q-table helper functions
└── README.md
```

## 🧠 Implemented Algorithms

### Value-Based Methods
- **Q-Learning**: Off-policy temporal difference learning with ε-greedy exploration.
- **SARSA**: On-policy temporal difference learning.
- **Double Q-Learning**: Reduces overestimation bias in Q-Learning.
- **Dynamic Programming**: Model-based planning approach.

### Policy-Based Methods
- **Temporal Difference (TD)**: Direct value function learning.

### Multi-Armed Bandits
- **Simple Bandit**: ε-greedy action selection for stateless environments.

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/dzungphieuluuky/RL_project.git
cd RL_project
```

2. Install dependencies:
```bash
pip install numpy torch gymnasium
```

## 💡 Usage

### Basic Q-Learning Example

```python
from agents.q_learning import QLearningAgent
import gymnasium as gym

# Create environment
env = gym.make('FrozenLake-v1')

# Initialize agent
agent = QLearningAgent(
    env=env,
    alpha=0.1,      # Learning rate / Step size
    gamma=0.99,     # Discount factor
    epsilon=1.0,    # Exploration rate
    epsilon_decay=0.995
)

# Train the agent
agent.train(num_episodes=1000)

# Save the learned Q-table
agent.save_table('q_table.npy')
```

### SARSA Agent

```python
from agents.sarsa import SARSAAgent

agent = SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
# Training logic here...
```

### Multi-Armed Bandit

```python
from agents.simple_bandit import SimpleBandit

bandit = SimpleBandit(env, actions=10, epsilon=0.1)
bandit.play(num_episodes=1000)
```

## 🔧 Key Features

### Configurable Hyperparameters
All agents support customizable learning rates, discount factors, and exploration strategies.

### Automatic Convergence Detection
Q-Learning agent includes threshold-based training termination for optimal learning.

### Model Persistence
Save and load trained models for later use:
```python
agent.save_table('model.npy')
agent.load_table('model.npy')
```

### Utility Functions
Helper functions for Q-table operations:
- [`get_best_action`](utils/getter.py): Find optimal action for a given state
- [`get_best_value`](utils/getter.py): Get maximum Q-value for a state

## 🎯 Algorithm Details

### Q-Learning
- **Type**: Off-policy, model-free
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **Exploration**: ε-greedy with decay

### SARSA
- **Type**: On-policy, model-free
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
- **Exploration**: ε-greedy

### Double Q-Learning
- **Advantage**: Reduces overestimation bias
- **Method**: Maintains two Q-tables, alternately updated

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 🎓 Educational Use

This project is designed for personal practice and closely follows the algorithms presented in Sutton & Barto's textbook. Each agent includes clear, well-commented code that makes it easy to understand the underlying mathematical concepts.

---

*Built with ❤️ for the RL community*