def get_best_action(q_table, state):
    return max(q_table[state], key=q_table[state].get)

def get_best_value(q_table, state):
    return max(q_table[state])