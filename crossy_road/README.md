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

### Libraries

The project dependencies, also listed in `requirements.txt`, are:

| Library | Purpose |
| --- | --- |
| [Pygame](https://www.pygame.org/) (`pygame>=2.5`) | Window, input, rendering, sprites, and game loop |
| [NumPy](https://numpy.org/) (`numpy>=1.26`) | Observation batches and saving/loading model weights |
| [PyTorch](https://pytorch.org/) (`torch>=2.2`) | Q-value neural network, automatic differentiation, and training |

The code also uses Python standard-library modules such as `random`, `enum`,
`pathlib`, and `os`; these do not need to be installed separately. PyTorch
selects CUDA when available, Apple MPS on supported Macs, or CPU otherwise.

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

## Configuration: `constants.py`

The main game and demo settings live in `crossy_road/constants.py`. Edit these
values to adjust the game; some of the values are defaults or metadata and are
not currently used by the runtime (noted below).

| Setting | Default | Effect |
| --- | --- | --- |
| `WIN_WIDTH`, `WIN_HEIGHT` | Detected display width and 90% of display height | Window and world dimensions |
| `FPS` | `120` | Target render rate for the standalone demo |
| `AI_UPDATE_SPEED` | `0.125` | Updates per rendered frame; roughly one agent update every eight frames |
| `TRAINING_GAMES_X`, `TRAINING_GAMES_Y` | `3`, `1` | Demo arena layout; all arenas share one model |
| `LANE_HEIGHT` | `80` pixels | Height of one movement tile/lane |
| `NUM_LANES` | `WIN_HEIGHT // LANE_HEIGHT` | Derived visible lane count; not a fixed map limit |
| `PLAYER_SIZE`, `PLAYER_RENDER_SIZE` | `50`, `LANE_HEIGHT` pixels | Nominal player size and rendered sprite size; rendering uses `PLAYER_RENDER_SIZE` |
| `PLAYER_COLOR` | `(255, 215, 0)` | Fallback player color if player artwork is unavailable |
| `PLAYER_ART_SMOOTHING` | `0.28` | How quickly the rendered character follows its logical position |
| `PLAYER_HITBOX_SCALE` | `0.8` | Collision hitbox size relative to the player artwork |
| `CAR_MIN_WIDTH`, `CAR_MAX_WIDTH` | `80`, `180` pixels | Fallback car width bounds; sprite artwork determines width when loaded |
| `CAR_MIN_SPEED`, `CAR_MAX_SPEED` | `8`, `8` | Default-biome car speed range |
| `CAR_SPEED` | `8` | Default speed used by a car when no explicit speed is passed |
| `TRAFFIC_START_LANES_ABOVE_CAMERA` | `-1` | Traffic spawn-area offset around the camera/player |
| `TRAFFIC_AMOUNT` | `20.0` | Multiplier on traffic-row selection weights and spawn cooldown |
| `TRAIN_PROBABILITY` | `0.12` | Default-biome train probability metadata; row selection currently uses `ROW_TYPES` weights instead |
| `TRAIN_SPEED` | `20` | Fixed train speed |
| `TRAIN_SPEED_FACTOR` | `0.8` | Declared train-speed tuning value; currently unused (trains use `TRAIN_SPEED`) |
| `MAX_LIVES` | `1` | Player lives setting |
| `FOLLOW_DISTANCE` | `2 * LANE_HEIGHT` | Camera's target vertical offset from the player |
| `CAMERA_SMOOTHING` | `0.5` | Camera easing toward its target offset |
| `BIOME_CHANGE_EVERY` | `15 * LANE_HEIGHT` | World distance between biome changes |
| `ROW_TYPES` | Grass `0.40`, car roads `0.20` each, train tracks `0.10` each | Relative weights used to generate map rows; traffic row weights are multiplied by `TRAFFIC_AMOUNT` |
| `BIOMES` | Default, Desert, Forest, City | Per-biome terrain colors and car speed ranges |
| `VIEW_LANES_AHEAD` | `5` | How many lanes ahead are included in the detailed obstacle list |
| `SENSING_GRID_ROWS`, `SENSING_GRID_COLUMNS` | `5`, `5` | Dimensions of the model's local sensing grid |
| `SENSING_RADIUS_TILES`, `SENSING_SIZE_TILES` | Derived from the sensing grid | Additional observation metadata exposed by the API |
| `BG_COLOR` | `(0, 0, 0)` | Rendered window background color |

Car speeds increase by approximately 10% every 50 lanes. The biome dictionaries
also include `train_prob` values, but train/road row selection is currently
controlled by `ROW_TYPES` and its relative weights.

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
kept as targets for that update. The reward is **+1** when the score increases,
**-2** when it does not, and **-10000** for a collision/death. The discount
factor `gamma` is `0.6`.

On each decision the agent explores by choosing a random action 30% of the
time; otherwise it chooses the action with the highest predicted Q-value. For
each update, the demo gathers transitions from the active arenas, predicts
their current and next Q-values, and trains on the batch using mean-squared
error and gradient descent. The learning rate is `0.01`, and gradients are
clipped for stability. This is online learning: the implementation does not
use a replay buffer or a separate target network.

### Model settings: `model.py`

The following module-level values configure the network and its learning:

| Setting | Default | Effect |
| --- | --- | --- |
| `load` | `True` | Load a compatible saved checkpoint when the model module starts |
| `learning_rate` | `0.01` | Gradient-descent step size |
| `gamma` | `0.6` | Weight given to the best next-state Q-value |
| `exploration_rate` | `0.3` | Probability of selecting a random action rather than the current best |
| `REWARD_SCORE_INCREASE` | `1` | Reward when the score rises |
| `REWARD_NO_PROGRESS` | `-2` | Reward when the score does not rise |
| `REWARD_DEATH` | `-10000` | Reward assigned for collision/death |
| `INPUTS_PER_CELL` | `2` | Terrain and vehicle codes per sensing cell |
| `outputs` | `5` | Q-values/actions: left, right, up, down, and no-op |
| `layers` | `3` | Number of affine layers: input-to-hidden, hidden-to-hidden, and hidden-to-output |
| `neurons` | `32` | Units in each hidden layer |
| `LEAKY_RELU_SLOPE` | `0.01` | Negative-side slope for hidden-layer activations |
| `MAX_GRADIENT_NORM` | `1.0` | Gradient clipping threshold |
| `MAX_PARAMETER_MAGNITUDE` | `5.0` | Bound used when sanitizing weights and biases |
| `WEIGHT_DECIMALS` | `6` | Precision used to round parameters |
| `SAVE_FILE` | `model_weights.npz` | Checkpoint path, relative to the process's current working directory |

`SENSING_CELL_COUNT` is calculated from the grid dimensions, and `IL` is
calculated as the cell count times `INPUTS_PER_CELL` (50 scalar inputs for the
default 5-by-5 grid). `weights` and `biases` hold the trainable PyTorch
parameters; `inputs` and `q` are scratch values for the latest input and
predicted outputs. `DEVICE` and `DEVICE_NAME` indicate the selected compute
backend. The module-level `done` value is currently initialized to `0` and is
not updated for each transition, so the Q-learning target does not mask the
next-state value on terminal transitions.

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
