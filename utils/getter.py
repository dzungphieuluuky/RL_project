def get_best_action(q_table, state):
    best_value = None
    best_action = 0
    for current_state, current_action in q_table:
        if current_state != state:
            continue
        if best_value is None or best_value < q_table[(current_state, current_action)]:
            best_value = q_table[(current_state, current_action)]
            best_action = current_action
    return best_action
        
def get_best_value(q_table, state):
    return max(q_table[state])