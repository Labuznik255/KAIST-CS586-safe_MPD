# `tests/test_planners/test_base.py`

**Source file:** [tests/test_planners/test_base.py](../../../tests/test_planners/test_base.py)

## Overview
Provides `BaseMBDTest` (the `unittest.TestCase` base class for planner integration tests) and `TimeoutContext`. Handles environment setup, timeout enforcement, goal-distance validation, trajectory safety checking, and JSON/NPZ artifact saving under `tests/test_planners/results/`.

---

## Class: `BaseMBDTest(unittest.TestCase)` (lines 37–334)

### `setUpClass(cls)` (`@classmethod`, lines 40–49)
Suppresses verbose JAX XLA bridge warnings and ensures `tests/test_planners/results` exists.

### `setUp(self)` (lines 51–54)
Records `self.start_time` and logs test start.

### `tearDown(self)` (lines 56–60)
Logs elapsed time for the test method.

### `run_mbd_test(self, config: TestConfig) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]` (lines 65–123)
1. Builds `env = create_test_tt2d_environment(config)`.
2. Runs `run_diffusion(args=config, env=env)` inside `TimeoutContext(config.timeout_seconds)`.
3. Converts `actions` and `states` to NumPy arrays.
4. Computes `final_distance, tractor_distance, trailer_distance` via `compute_final_min_distance`.
5. Calls `validate_test_results(...)` and `save_test_results(...)`.
6. Returns `(final_distance, actions_np, states_np, timing)`.

### `validate_test_results(...)` (lines 125–165)
Asserts:
- `expected_distance_min <= final_distance <= expected_distance_max`.
- Logs trajectory hitch/collision violations (`check_trajectory_violations`) without hard-failing on violations.
- Asserts `position_success` (`check_position_success`).
- Checks array shapes: `len(actions) == config.Hsample`, `len(states) == config.Hsample + 1`, `actions.shape[1] == 2`.

### `check_trajectory_violations(self, states: np.ndarray, env) -> Tuple[bool, bool]` (lines 166–198)
Iterates through every state in `states`, slices `state_4d = state_jax[:4]`, and checks `env.check_hitch_violation(state_4d)` and `env.check_obstacle_collision(state_4d, env.obs_circles, env.obs_rectangles)`.

### `check_position_success(self, final_state: np.ndarray, config: TestConfig, env) -> Tuple[bool, float, float]` (lines 200–235)
Computes Euclidean distances from `final_state[:2]` and `env.get_trailer_back_position(state_jax[:4])` to `env.xg[:2]`, returning `(success, tractor_distance, trailer_distance)` where `success` requires at least one to be `<= config.min_final_distance_to_goal`.

### `compute_final_min_distance(self, states: np.ndarray, env) -> Tuple[float, float, float]` (lines 237–256)
Computes `(min(tractor_distance, trailer_distance), tractor_distance, trailer_distance)` at `states[-1][:4]`.

### `save_test_results(...)` (lines 258–312)
Writes `{test_name}_result.json` (config dict, final distance/state, timing, timestamp, git commit, summary) and `{test_name}_arrays.npz` (`actions`, `states`) to `self.test_results_dir`.

### `get_git_commit(self) -> str` (lines 314–325)
Runs `git rev-parse HEAD` and returns the hash (or `"unknown"`).

### `get_system_info(self) -> Dict[str, str]` (lines 327–334)
Returns Python version, OS platform, and processor string.

---

## Class: `TimeoutContext` (lines 337–352)
Context manager (`__enter__`, `__exit__`) checking elapsed wall-clock time on exit and raising `TimeoutError` if `elapsed > timeout_seconds` (note: post-hoc check on block exit, not preemptive signal interruption).

---

## Extension Notes
- **New vehicle class**:
  - `validate_test_results` (line 164) asserts `actions.shape[1] == 2`! If your new vehicle has `Nu != 2`, update that assertion (e.g., compare against `env.action_size`).
  - `check_trajectory_violations`, `check_position_success`, and `compute_final_min_distance` slice `state[:4]`, assuming `[px, py, theta1, theta2]` occupy the first 4 indices and `env.get_trailer_back_position` exists.
