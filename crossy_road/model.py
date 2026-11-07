import numpy as np
from . import API
learning_rate = 0.01
gamma = 0.1
done = 0

def bellman(r,q):
    Q = r + gamma * max(q) * (1 - done)
    return Q, q


inputs = np.array([1,2,3,4])
IL = len(inputs)
outputs = 5
layers = 5
neurons = 8

# CHANGED: Initialize each layer separately
weights = [
    np.random.randn(neurons, IL)
    for _ in range(layers - 1)
]

weights.append(np.random.randn(outputs, neurons))

# CHANGED: Initialize biases separately
biases = [
    np.zeros(neurons)
    for _ in range(layers - 1)
]

biases.append(np.zeros(outputs))


def model(A,L=layers):  # CHANGED: Added default layer count
    for l in range(L):
        Z = weights[l] @ A + biases[l]
        A = np.maximum(0,Z) if l < layers-1 else Z  # CHANGED: No ReLU on output
    return A


# Derivatives:
dW = [np.zeros_like(w) for w in weights]
dB = [np.zeros_like(b) for b in biases]

# NEW: Store activations for backpropagation
def train():
    activations = [inputs]
    for l in range(layers):
        Z = weights[l] @ activations[-1] + biases[l]
        A = np.maximum(0,Z) if l < layers-1 else Z
        activations.append(A)

    actual = bellman(r,q)[0]
    predicted = model(inputs)

    # CHANGED: Calculate squared error gradient
    dL = 2 * (predicted - actual)

    # NEW: Backpropagation
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
    position = data["player_position"]
    sense = data["sensing_grid"]
    I = position + sense
    return I

def getReward(data):
    score = data["current_score"]
    return score

def getQ(data):

    obstacles = data["nearby_obstacles"]
    Qs = np.zeros[len(obstacles),len(obstacles[0])]
    for x in obstacles:
        for y in obstacles[x]:
            obstacle = obstacles[x][y]
            if obstacle == 0 or obstacle == 1:
                Qs[x][y] = 1
            else:
                Qs[x][y] = 0
    return Qs


def update(env):
    """Run once after each game tick and return the current player data."""
    global inputs,r,q
    data = API.get_player_data(env)
    inputs = getInputs(data)
    r = getReward(data)
    q = getQ(data)
    API.move_player(np.argmax(np.array(train())))
    return API.get_player_data(env)
