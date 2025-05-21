import random

class BaseAgent:
    def __init__(self, env):
        self.env = env
        pass

    def select_action(self):
        return random.sample(self.actions)

    def save(self, path):
        pass

    def load(self, path):
        pass

    