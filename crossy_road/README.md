# Crossy Road Remake

A Python/Pygame remake of Crossy Road with procedurally generated traffic and a
reinforcement-learning agent. The demo runs three game instances side by side;
they share one neural-network policy and train it as they play.

## Requirements and setup

Python 3.10 or newer is recommended. From the repository root (the directory
containing `crossy_road/`), create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r crossy_road/requirements.txt
```

The dependencies are Pygame, NumPy, and PyTorch. PyTorch uses CUDA when
available, Apple MPS on supported Macs, or CPU otherwise.

## Run the demo

From the repository root, with the environment activated:

```bash
python -m crossy_road.run_demo
```

Press **Space** at the menu to start. The three arenas then run and train the
shared agent. Press **P** to pause or resume. While paused, press **R** to reset
all arenas and start again. Close the window to quit.

The arrow keys submit a move to the arena beneath the mouse pointer. The
agent continues to make its own moves and learn in all active arenas.

Each arena also has three buttons:

- **Grid** toggles the tile grid, the agent's sensing area, and a visualization
  of action scores and input effects.
- **Impact** shows how input features affect the currently highest-scoring
  action. Green indicates a positive effect and red a negative one; scroll over
  the arena to inspect more entries.
- **Step** requests an agent update for that arena.

The HUD shows the current score, high score, and the model's five action
outputs. The player advances one lane per successful forward move. A collision
resets that arena's player and map; the high score is retained. The map is
generated as grass, car roads, and train tracks, with traffic moving in either
direction. Scenery and traffic settings change across biomes, and traffic gets
faster as the player progresses. Character and vehicle artwork is loaded from
`assets/` when available, with simple drawn shapes used as fallbacks.

## Test the project

There is no separate automated test suite in the project. From the repository
root, first check Python syntax:

```bash
python -m compileall -q crossy_road
```

Then run a small environment/API/model smoke test:

```bash
python - <<'PY'
from crossy_road import Action, Game
from crossy_road import API, model

game = Game(render=False)
game.reset(seed=7)
data = API.get_player_data(game)
inputs = model.getInputs(data)
assert model.predict(inputs).shape == (5,)
observation, reward, done, info = game.step(Action.NOOP)
assert "player" in observation and "collision" in info
print("Environment, observation, model, and step are working.")
PY
```

`Game.reset(seed=...)` seeds procedural map generation for repeatable
single-environment checks. To run the graphical demo as an end-to-end check,
use the launch instructions above.

## Agent and training algorithm

The agent is a small, fully connected Q-value network implemented in PyTorch.
It maps the local game state to a score (Q-value) for each of five actions:
`LEFT`, `RIGHT`, `UP`, `DOWN`, and `NOOP`.

### Observation and network

The API constructs a 5-by-5 sensing grid centered around the player. Every cell
is encoded by two numeric features:

- **Terrain:** `0` for land or unknown ground, `1` for a car road, and `2` for
  train tracks.
- **Vehicle:** `0` for none, `1` for a car, or `2` for a train.

Flattening the two features for all 25 cells gives the network 25 feature pairs
(50 numeric inputs). The network has two 32-unit hidden layers with Leaky ReLU
activations and a five-value output layer. The highest output is the action the
policy currently prefers.

### Learning loop

The training is online, one-step Q-learning, using a Bellman-style target:

```text
target(action) = reward + gamma * max(Q(next_state))
```

Only the selected action's target is changed; the other current Q-values are
kept as targets for that update. The reward is **+5** when the score increases,
**-1** when it does not, and **-2000** for a collision/death. The discount
factor `gamma` is `0.9`.

On each decision the agent explores by choosing a random action 30% of the
time; otherwise it chooses the action with the highest predicted Q-value. For
each update, the demo gathers transitions from the active arenas, predicts
their current and next Q-values, and trains on the batch using mean-squared
error and gradient descent. The learning rate is `0.01`, and gradients are
clipped for stability. This is online learning: the implementation does not
use a replay buffer or a separate target network.

The model loads `model_weights.npz` from the process's current working
directory, if a compatible file exists, and saves updated weights to that file
after each training batch. Start the demo consistently from the repository root
to keep using the same checkpoint location. The checked-in weights in
`crossy_road/model_weights.npz` are not automatically selected when running
from the repository root; the model looks in the current working directory.

## Python API

The game can also be stepped without opening a window:

```python
from crossy_road import Action, Game

game = Game(render=False)
observation = game.reset(seed=7)
observation, reward, done, info = game.step(Action.UP)
```

`Game.step()` returns the latest observation, reward, completion flag, and
diagnostic information. `Game.get_observation()` exposes player position,
score, legal actions, generated map rows, and obstacle data. The `crossy_road.API`
module provides `get_player_data(env)` for the sensing-grid observation and
`move_player(env, direction)` for submitting an `Action`, direction string, or
five Q-values. The game loop does not train the model on its own; call
`crossy_road.model.update(env)` (or `update_many(envs)`) to run an agent
decision and training update.
