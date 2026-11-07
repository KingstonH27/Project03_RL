import random

import numpy as np
from . import API

learning_rate = 0.01
gamma = 0.1
done = 0
exploration_rate = 0.1
REWARD_SCORE_INCREASE = 100
REWARD_NO_PROGRESS = -1
REWARD_DEATH = -100

def bellman(r,q):
    Q = r + gamma * max(q) * (1 - done)
    return Q, q

IL = 27
inputs = np.zeros(IL)
outputs = 5
layers = 2
neurons = 4

weights = [
    np.random.randn(neurons, IL)
]

weights.extend([
    np.random.randn(neurons, neurons)
    for _ in range(layers - 2)
])

weights.append(np.random.randn(outputs, neurons))


biases = [
    np.zeros(neurons)
    for _ in range(layers - 1)
]

biases.append(np.zeros(outputs))

import os

SAVE_FILE = "model_weights.npz"
def save_model():
    np.savez(
        SAVE_FILE,
        **{f"w{i}": w for i, w in enumerate(weights)},
        **{f"b{i}": b for i, b in enumerate(biases)}
    )


def load_model():
    if os.path.exists(SAVE_FILE):
        data = np.load(SAVE_FILE)

        for i in range(len(weights)):
            weights[i] = data[f"w{i}"]
            biases[i] = data[f"b{i}"]

        print("Model loaded!")
    else:
        print("No saved model found. Using random weights.")





load_model()


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


def getReward(data, score_before, step_info):
    if step_info.get("collision", False) or not data["player"]["alive"]:
        return REWARD_DEATH
    if data["current_score"] > score_before:
        return REWARD_SCORE_INCREASE
    return REWARD_NO_PROGRESS


def getQ(reward, future_q, current_q, action):
    actual = current_q.copy()
    actual[action] = reward + gamma * max(future_q) * (1 - done)
    return actual


def update(env):
    global inputs, r, q

    # Get current state
    data = API.get_player_data(env)
    inputs = np.array(getInputs(data))
    #print("Inputs:", inputs.shape)
    #print("Weights:", weights[0].shape)
    #print("Player position:", np.array(data["player_position"]).shape)
    #print("Sensing grid:", np.array(data["sensing_grid"]).shape)

    # Predict current Q-values
    q = model(inputs)

    legal_action_indices = [
        index
        for index, candidate in enumerate(API._DIRECTION_ACTIONS)
        if data["legal_actions"].get(candidate.name, False)
    ]

    if random.random() < exploration_rate:
        action = random.choice(legal_action_indices)
    else:
        action = max(legal_action_indices, key=lambda index: q[index])

    # Execute action
    score_before = data["current_score"]
    _, _, _, step_info = env.step(API._DIRECTION_ACTIONS[action])

    # Get next state
    next_data = API.get_player_data(env)
    next_inputs = np.array(getInputs(next_data))

    # Predict future Q-values
    future_q = model(next_inputs)

    reward = getReward(next_data, score_before, step_info)

    actual = getQ(reward, future_q, q, action)

    train(actual)
    save_model()

    # Update inputs
    inputs = next_inputs

    return next_data