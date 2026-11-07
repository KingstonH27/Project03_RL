learningRate = 0.1


def bellman():
    target = reward + learningRate * max(next_q_values) * (1 - done)
    return target