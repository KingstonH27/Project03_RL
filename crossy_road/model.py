import random

import numpy as np
from . import API
from .constants import (
    LANE_HEIGHT,
    SENSING_GRID_COLUMNS,
    SENSING_GRID_ROWS,
    WIN_WIDTH,
)
load = True
learning_rate = 0.1
gamma = 0.5
done = 0
exploration_rate = 0.3
REWARD_SCORE_INCREASE = 1
REWARD_NO_PROGRESS = -5
REWARD_DEATH = -100

def bellman(r,q):
    Q = r + gamma * max(q) * (1 - done)
    return Q, q

POSITION_INPUTS = 2
SENSING_CELL_COUNT = SENSING_GRID_ROWS * SENSING_GRID_COLUMNS
TILE_FEATURE_COUNT = len(API.SENSING_TILE_TYPES)
VEHICLE_FEATURE_COUNT = len(API.SENSING_VEHICLE_TYPES)
IL = POSITION_INPUTS + SENSING_CELL_COUNT * (
    TILE_FEATURE_COUNT + VEHICLE_FEATURE_COUNT
)
inputs = np.zeros(IL)
outputs = 5
q = np.zeros(outputs)
layers = 3
neurons = 10
LEAKY_RELU_SLOPE = 0.01
MAX_GRADIENT_NORM = 1.0
MAX_PARAMETER_MAGNITUDE = 5.0
WEIGHT_DECIMALS = 6

weights = [
    np.round(np.random.rand(neurons, IL) / IL, WEIGHT_DECIMALS)
]

weights.extend([
    np.round(np.random.rand(neurons, neurons) / neurons, WEIGHT_DECIMALS)
    for _ in range(layers - 2)
])

weights.append(
    np.round(np.random.rand(outputs, neurons) / neurons, WEIGHT_DECIMALS)
)


biases = [
    np.zeros(neurons)
    for _ in range(layers - 1)
]

biases.append(np.zeros(outputs))

import os

SAVE_FILE = "model_weights.npz"


def _stabilize_parameters(values):
    finite_values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=MAX_PARAMETER_MAGNITUDE,
        neginf=-MAX_PARAMETER_MAGNITUDE,
    )
    bounded_values = np.clip(
        finite_values,
        -MAX_PARAMETER_MAGNITUDE,
        MAX_PARAMETER_MAGNITUDE,
    )
    return np.round(bounded_values, WEIGHT_DECIMALS)


def save_model():
    np.savez(
        SAVE_FILE,
        **{f"w{i}": w for i, w in enumerate(weights)},
        **{f"b{i}": b for i, b in enumerate(biases)}
    )


def load_model():
    if os.path.exists(SAVE_FILE):
        data = np.load(SAVE_FILE)

        for index, (weight, bias) in enumerate(zip(weights, biases)):
            if (
                data[f"w{index}"].shape != weight.shape
                or data[f"b{index}"].shape != bias.shape
            ):
                print("Saved model shape differs from this network; using fresh weights.")
                return

        for i in range(len(weights)):
            weights[i] = _stabilize_parameters(data[f"w{i}"])
            biases[i] = _stabilize_parameters(data[f"b{i}"])

        print("Model loaded!")
    else:
        print("No saved model found. Using random weights.")




if load:
    load_model()


def _leaky_relu(values):
    return np.where(values > 0, values, LEAKY_RELU_SLOPE * values)


def _leaky_relu_derivative(activations):
    return np.where(activations > 0, 1.0, LEAKY_RELU_SLOPE)


def model(A,L=layers):
    for l in range(L):
        Z = weights[l] @ A + biases[l]
        A = _leaky_relu(Z) if l < layers-1 else Z
    return A


def input_sensitivities(input_values, output_index):
    activations = [np.asarray(input_values, dtype=float)]

    for layer_index in range(layers):
        Z = weights[layer_index] @ activations[-1] + biases[layer_index]
        activation = (
            _leaky_relu(Z)
            if layer_index < layers - 1
            else Z
        )
        activations.append(activation)

    gradient = np.zeros(outputs)
    gradient[output_index] = 1.0

    for layer_index in range(layers - 1, -1, -1):
        if layer_index < layers - 1:
            gradient *= _leaky_relu_derivative(activations[layer_index + 1])
        gradient = weights[layer_index].T @ gradient

    return gradient


# Derivatives:
dW = [np.zeros_like(w) for w in weights]
dB = [np.zeros_like(b) for b in biases]


def train(actual):  # CHANGED: Accept five target Q-values
    activations = [inputs]
    for l in range(layers):
        Z = weights[l] @ activations[-1] + biases[l]
        A = _leaky_relu(Z) if l < layers-1 else Z
        activations.append(A)

    predicted = model(inputs)

    dL = 2 * (predicted - actual)

    # Backpropagation
    for l in range(layers - 1, -1, -1):
        dZ = (
            dL * _leaky_relu_derivative(activations[l + 1])
            if l < layers - 1
            else dL
        )

        dW[l] = np.outer(dZ, activations[l])
        dB[l] = dZ

        dL = weights[l].T @ dZ

    gradients = dW + dB
    for gradient in gradients:
        np.nan_to_num(
            gradient,
            copy=False,
            nan=0.0,
            posinf=MAX_GRADIENT_NORM,
            neginf=-MAX_GRADIENT_NORM,
        )

    max_gradient = max(
        (float(np.max(np.abs(gradient))) for gradient in gradients),
        default=0.0,
    )
    if max_gradient > 0.0:
        scaled_norm = np.sqrt(sum(
            np.sum((gradient / max_gradient) ** 2)
            for gradient in gradients
        ))
        gradient_scale = min(
            1.0,
            (MAX_GRADIENT_NORM / max_gradient) / scaled_norm,
        )
    else:
        gradient_scale = 1.0

    for l in range(layers):
        weights[l] = _stabilize_parameters(
            weights[l] - learning_rate * gradient_scale * dW[l]
        )
        biases[l] = _stabilize_parameters(
            biases[l] - learning_rate * gradient_scale * dB[l]
        )

    return model(inputs)


def getInputs(data):
    position = np.asarray(data["player_position"], dtype=float).flatten()
    position[0] = np.clip(position[0] / WIN_WIDTH, 0.0, 1.0)
    position[1] = np.clip(position[1] / (100 * LANE_HEIGHT), 0.0, 1.0)

    tiles = np.asarray(data["sensing_tiles"]).reshape(-1)
    tile_indices = {
        tile_type: index
        for index, tile_type in enumerate(API.SENSING_TILE_TYPES)
    }
    tile_features = np.zeros((SENSING_CELL_COUNT, TILE_FEATURE_COUNT))
    for cell_index, tile_type in enumerate(tiles):
        tile_features[cell_index, tile_indices[tile_type]] = 1.0

    vehicle_features = np.asarray(
        data["sensing_vehicles"],
        dtype=float
    ).reshape(SENSING_CELL_COUNT, VEHICLE_FEATURE_COUNT)

    return np.concatenate((
        position,
        tile_features.flatten(),
        vehicle_features.flatten(),
    ))


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
    #print("Sensing tiles:", np.array(data["sensing_tiles"]).shape)
    #print("Sensing vehicles:", np.array(data["sensing_vehicles"]).shape)

    # Predict current Q-values
    q = model(inputs)

    action_indices = range(len(API._DIRECTION_ACTIONS))

    if random.random() < exploration_rate:
        action = random.choice(action_indices)
    else:
        action = max(action_indices, key=lambda index: q[index])

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
    q = model(inputs)

    return next_data