import random

class BaseAgent:
    def __init__(self, env):
        self.env = env
        pass

    def select_action(self):
        return random.sample(self.actions)

    def update(self):
        pass

    