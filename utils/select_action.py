import numpy as np

def random_selection(actions):
    return np.random.random(actions)

def epsilon_selection(q_value, actions, state, epsilon=0.7):
    if np.random.randint(0, 1) < epsilon:
        return random_selection(actions)
    else:
        return np.argmax(q_value[state])