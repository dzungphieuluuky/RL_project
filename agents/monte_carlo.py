import numpy as np

from .base_agent import BaseAgent

class MCPrediction(BaseAgent):
    def __init__(self, env, gamma = 0.99):
        super().__init__(env)
        self.gamma = gamma
        self.action_space = env.action_space.n
        self.observation_space = env.observation_space.n
        self.policy = np.random.random((self.observation_space, self.action_space))

        # normalize policy
        self.policy = self.policy / self.policy.sum(axis=1, keepdims=True)

        self.values = np.zeros(self.observation_space)
        self.returns = [[] for _ in range(self.observation_space)]
    
    def evaluate(self, num_episodes=1000, first_visit = False):
        for _ in range(num_episodes):
            t = 0
            state = self.env.reset()[0]
            done = False
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_rewards.append(0)

            while not done:
                action = np.random.choice(self.action_space, p=self.policy[state])
                next_state, reward, done, truncate, info = self.env.step(action)
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                state = next_state
                t += 1

                if done or truncate:
                    break
            
            g = 0
            t -= 1

            for i in reversed(range(t)):
                g = self.gamma * g + episode_rewards[i + 1]
                current_state = episode_states[i]

                if first_visit and current_state in episode_states[:i]:
                    continue
                else:
                    self.returns[current_state].append(g)
                    self.values[current_state] = np.mean(self.returns[current_state])

