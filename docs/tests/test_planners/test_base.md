# `tests/test_planners/test_base.py`

**Source file:** [tests/test_planners/test_base.py](../../../tests/test_planners/test_base.py)

## Overview
Provides `BaseMBDTest` (the `unittest.TestCase` base class for planner integration tests) and `TimeoutContext`. Handles environment setup, timeout enforcement, goal-distance validation, trajectory safety checking, and JSON/NPZ artifact saving (including both `nominal_states` and `perturbed_states`) under `tests/test_planners/results/`.

---

## Class: `BaseMBDTest(unittest.TestCase)` (lines 37–342)

- **`setUpClass(cls)`** (`@classmethod`, lines 41–49), **`setUp(self)`** (lines 51–54), **`tearDown(self)`** (lines 56–60).
- **`run_mbd_test(self, config: TestConfig)`** (lines 65–123): Creates environment via `create_test_tt2d_environment(config)`, runs `run_diffusion(args=config, env=env)` inside `TimeoutContext`, validates results, and saves outputs.
- **`validate_test_results(...)`** (lines 125–164): Checks distance bounds, logs trajectory violations, asserts `position_success`, and verifies array shapes.
- **`check_trajectory_violations(self, states: np.ndarray, env)`** (lines 166–198): Checks `check_hitch_violation` and `check_obstacle_collision` across `states`.
- **`check_position_success(self, final_state: np.ndarray, config: TestConfig, env)`** (lines 200–235) & **`compute_final_min_distance(self, states: np.ndarray, env)`** (lines 237–256): Compute tractor and trailer distances to goal.
- **`save_test_results(...)`** (lines 258–320): Writes `{test_name}_result.json` and saves `actions`, `states`, `nominal_states`, and `perturbed_states` to `{test_name}_arrays.npz`.
- **`get_git_commit(self)`** (lines 322–333) & **`get_system_info(self)`** (lines 335–342).

## Class: `TimeoutContext` (lines 345–360)
Context manager raising `TimeoutError` if elapsed time exceeds `timeout_seconds`.
