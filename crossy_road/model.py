import numpy as np
from . import API

learning_rate = 0.01
gamma = 0.1
done = 0
previous_score = 0  # NEW

def bellman(r,q):
    Q = r + gamma * max(q) * (1 - done)
    return Q, q

IL = 27
inputs = np.zeros(IL)
outputs = 5
layers = 5
neurons = 8

weights = [
    np.random.randn(neurons, IL)
]

weights.extend([
    np.random.randn(neurons, neurons)
    for _ in range(layers - 2)
])

weights.append(np.random.randn(outputs, neurons))

weights.append(np.random.randn(outputs, neurons))

biases = [
    np.zeros(neurons)
    for _ in range(layers - 1)
]

biases.append(np.zeros(outputs))


def model(A,L=layers):
    for l in range(L):
        Z = weights[l] @ A + biases[l]
        A = np.maximum(0,Z) if l < layers-1 else Z
    return A


# Derivatives:
dW = [np.zeros_like(w) for w in weights]
dB = [np.zeros_like(b) for b in biases]


def train(actual):  # CHANGED: Accept five target Q-values
    activations = [inputs]
    for l in range(layers):
        Z = weights[l] @ activations[-1] + biases[l]
        A = np.maximum(0,Z) if l < layers-1 else Z
        activations.append(A)

    predicted = model(inputs)

    dL = 2 * (predicted - actual)

    # Backpropagation
    for l in range(layers - 1, -1, -1):
        dZ = dL * (activations[l+1] > 0) if l < layers-1 else dL

        dW[l] = np.outer(dZ, activations[l])
        dB[l] = dZ

        dL = weights[l].T @ dZ

    # Update Weights:
    for l in range(layers):
        weights[l] -= learning_rate * dW[l]
        biases[l] -= learning_rate * dB[l]

    return model(inputs)


def getInputs(data):
    position = np.array(data["player_position"]).flatten()
    sense = np.array(data["sensing_grid"]).flatten()
    return np.concatenate((position, sense))


def getReward(data):
    global previous_score  # NEW

    score = data["current_score"]
    reward = 1 if score > previous_score else 0
    previous_score = score  # NEW

    return reward


def getQ(data, future_q):
    obstacles = np.array(data["sensing_grid"]).reshape(5, 5)

    positions = [
        obstacles[2, 1],
        obstacles[2, 3],
        obstacles[1, 2],
        obstacles[3, 2],
        obstacles[2, 2]
    ]

    r = getReward(data)
    Qs = []

    for position in positions:
        reward = r if position in [0, 3] else -1
        Q = bellman(reward, future_q)[0]
        Qs.append(Q)

    return np.array(Qs)


def update(env):
    global inputs, r, q, previous_score

    # Get current state
    data = API.get_player_data(env)
    inputs = np.array(getInputs(data))
    print("Inputs:", inputs.shape)
    print("Weights:", weights[0].shape)
    print("Player position:", np.array(data["player_position"]).shape)
    print("Sensing grid:", np.array(data["sensing_grid"]).shape)

    # Predict current Q-values
    q = model(inputs)

    # Choose action
    action = np.argmax(q)

    # Execute action
    env.step(API._DIRECTION_ACTIONS[action])

    # Get next state
    next_data = API.get_player_data(env)
    next_inputs = np.array(getInputs(next_data))

    # Predict future Q-values
    future_q = model(next_inputs)

    # Calculate Bellman targets
    actual = getQ(data, future_q)

    # Train
    train(actual)

    # Update inputs
    inputs = next_inputs

    return next_data