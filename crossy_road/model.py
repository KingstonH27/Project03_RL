import random

import numpy as np
import torch
from . import API
from .constants import (
    SENSING_GRID_COLUMNS,
    SENSING_GRID_ROWS,
)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)
elif (
    getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
):
    DEVICE = torch.device("mps")
    DEVICE_NAME = "Apple MPS"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"
load = True
learning_rate = 0.1
gamma = 0.4
done = 0
exploration_rate = 0.1
REWARD_SCORE_INCREASE = 1
REWARD_NO_PROGRESS = -2
REWARD_DEATH = -10000

def bellman(r,q):
    Q = r + gamma * max(q) * (1 - done)
    return Q, q

SENSING_CELL_COUNT = SENSING_GRID_ROWS * SENSING_GRID_COLUMNS
INPUTS_PER_CELL = 2
IL = SENSING_CELL_COUNT * INPUTS_PER_CELL
inputs = np.zeros(IL)
outputs = 5
q = np.zeros(outputs)
layers = 2
neurons = 10
LEAKY_RELU_SLOPE = 0.01
MAX_GRADIENT_NORM = 1.0
MAX_PARAMETER_MAGNITUDE = 5.0
WEIGHT_DECIMALS = 6
PARAMETER_SCALE = 10 ** WEIGHT_DECIMALS
print(f"AI training device: {DEVICE_NAME} ({DEVICE})")

weights = [
    torch.nn.Parameter(
        torch.round(
            torch.rand((neurons, IL), device=DEVICE) / IL * PARAMETER_SCALE
        ) / PARAMETER_SCALE
    )
]

weights.extend([
    torch.nn.Parameter(
        torch.round(
            torch.rand((neurons, neurons), device=DEVICE)
            / neurons
            * PARAMETER_SCALE
        ) / PARAMETER_SCALE
    )
    for _ in range(layers - 2)
])

weights.append(
    torch.nn.Parameter(
        torch.round(
            torch.rand((outputs, neurons), device=DEVICE)
            / neurons
            * PARAMETER_SCALE
        ) / PARAMETER_SCALE
    )
)


biases = [
    torch.nn.Parameter(torch.zeros(neurons, device=DEVICE))
    for _ in range(layers - 1)
]

biases.append(torch.nn.Parameter(torch.zeros(outputs, device=DEVICE)))

import os

SAVE_FILE = "model_weights.npz"


def _stabilize_parameters(values):
    finite_values = torch.nan_to_num(
        values,
        nan=1.0,
        posinf=MAX_PARAMETER_MAGNITUDE,
        neginf=-MAX_PARAMETER_MAGNITUDE,
    )
    bounded_values = torch.clamp(
        finite_values,
        -MAX_PARAMETER_MAGNITUDE,
        MAX_PARAMETER_MAGNITUDE,
    )
    return torch.round(bounded_values * PARAMETER_SCALE) / PARAMETER_SCALE


def _sanitize_model_parameters():
    with torch.no_grad():
        for index in range(len(weights)):
            weights[index].copy_(_stabilize_parameters(weights[index]))
            biases[index].copy_(_stabilize_parameters(biases[index]))


def save_model():
    np.savez(
        SAVE_FILE,
        **{
            f"w{i}": weight.detach().cpu().numpy()
            for i, weight in enumerate(weights)
        },
        **{
            f"b{i}": bias.detach().cpu().numpy()
            for i, bias in enumerate(biases)
        }
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

        with torch.no_grad():
            for i in range(len(weights)):
                weights[i].copy_(torch.as_tensor(
                    data[f"w{i}"],
                    dtype=torch.float32,
                    device=DEVICE,
                ))
                biases[i].copy_(torch.as_tensor(
                    data[f"b{i}"],
                    dtype=torch.float32,
                    device=DEVICE,
                ))
        _sanitize_model_parameters()

        print("Model loaded!")
    else:
        print("No saved model found. Using random weights.")




if load:
    load_model()


def _leaky_relu(values):
    return torch.where(values > 0, values, LEAKY_RELU_SLOPE * values)


def model(A,L=layers):
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A, dtype=torch.float32, device=DEVICE)
    else:
        A = A.to(device=DEVICE, dtype=torch.float32)

    for l in range(L):
        Z = A @ weights[l].T + biases[l]
        A = _leaky_relu(Z) if l < layers-1 else Z
    return A


def predict(input_values):
    _sanitize_model_parameters()
    with torch.no_grad():
        return model(input_values).detach().cpu().numpy()


def input_sensitivities(input_values, output_index):
    _sanitize_model_parameters()
    input_tensor = torch.as_tensor(
        input_values,
        dtype=torch.float32,
        device=DEVICE,
    ).detach().requires_grad_(True)
    output = model(input_tensor)[output_index]
    gradient = torch.autograd.grad(output, input_tensor)[0]
    return gradient.detach().cpu().numpy()


def train(actual, input_values=None):
    _sanitize_model_parameters()
    training_inputs = inputs if input_values is None else input_values
    input_tensor = torch.as_tensor(
        training_inputs,
        dtype=torch.float32,
        device=DEVICE,
    )
    target_tensor = torch.as_tensor(
        actual,
        dtype=torch.float32,
        device=DEVICE,
    )
    parameters = weights + biases
    for parameter in parameters:
        parameter.grad = None

    loss = torch.nn.functional.mse_loss(
        model(input_tensor),
        target_tensor,
    )
    loss.backward()

    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.copy_(torch.nan_to_num(
                parameter.grad,
                nan=0.0,
                posinf=MAX_GRADIENT_NORM,
                neginf=-MAX_GRADIENT_NORM,
            ))

    torch.nn.utils.clip_grad_norm_(parameters, MAX_GRADIENT_NORM)

    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(parameter.grad, alpha=-learning_rate)
            parameter.copy_(_stabilize_parameters(parameter))
            parameter.grad = None

    return predict(input_tensor)


def getInputs(data):
    tiles = np.asarray(data["sensing_tiles"]).reshape(-1)
    terrain_codes = np.zeros(SENSING_CELL_COUNT, dtype=float)
    for cell_index, tile_type in enumerate(tiles):
        if tile_type in ("road_right", "road_left"):
            terrain_codes[cell_index] = 1.0
        elif tile_type in ("train_track_right", "train_track_left"):
            terrain_codes[cell_index] = 2.0

    vehicle_presence = np.asarray(
        data["sensing_vehicles"],
        dtype=float
    ).reshape(SENSING_CELL_COUNT, len(API.SENSING_VEHICLE_TYPES))
    vehicle_codes = np.zeros(SENSING_CELL_COUNT, dtype=float)
    car_index = API.SENSING_VEHICLE_TYPES.index("car")
    train_index = API.SENSING_VEHICLE_TYPES.index("train")
    vehicle_codes[vehicle_presence[:, car_index] > 0] = 1.0
    vehicle_codes[vehicle_presence[:, train_index] > 0] = 2.0

    return np.column_stack((terrain_codes, vehicle_codes)).flatten()


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


def update_many(envs):
    global inputs, q

    envs = list(envs)
    if not envs:
        return []

    current_data = [API.get_player_data(env) for env in envs]
    input_batch = np.stack([
        getInputs(data)
        for data in current_data
    ]).astype(np.float32)
    current_q = predict(input_batch)

    actions = []
    step_infos = []
    scores_before = []
    for env, data, action_values in zip(envs, current_data, current_q):
        if random.random() < exploration_rate:
            action_index = random.randrange(outputs)
        else:
            action_index = int(np.argmax(action_values))

        scores_before.append(data["current_score"])
        _, _, _, step_info = env.step(API._DIRECTION_ACTIONS[action_index])
        actions.append(action_index)
        step_infos.append(step_info)

    next_data = [API.get_player_data(env) for env in envs]
    next_input_batch = np.stack([
        getInputs(data)
        for data in next_data
    ]).astype(np.float32)
    future_q = predict(next_input_batch)

    targets = current_q.copy()
    for index, data in enumerate(next_data):
        reward = getReward(data, scores_before[index], step_infos[index])
        targets[index, actions[index]] = (
            reward + gamma * np.max(future_q[index]) * (1 - done)
        )

    train(targets, input_batch)
    save_model()

    inputs = next_input_batch[-1]
    q = predict(next_input_batch)[-1]
    return next_data


def update(env):
    return update_many([env])[0]