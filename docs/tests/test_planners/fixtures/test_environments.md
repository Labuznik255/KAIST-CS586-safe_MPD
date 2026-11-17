# `tests/test_planners/fixtures/test_environments.py`

**Source file:** [tests/test_planners/fixtures/test_environments.py](../../../../tests/test_planners/fixtures/test_environments.py)

## Overview
Provides helper functions to build customized `Env` and robot environments from a `TestConfig` for planner tests.

## Functions

### `setup_parking_environment(config: TestConfig) -> Env` (lines 27–55)
Constructs `Env(case="parking", parking_config=config.custom_parking_config or None)` and appends `config.custom_circular_obstacles` and `config.custom_rectangular_obstacles`.

### `create_test_tt2d_environment(config: TestConfig) -> Any` (lines 58–165)
1. Calls `setup_parking_environment(config)` to create `env_config`.
2. Calls `mbd.envs.get_env(config.env_name, ...)` forwarding geometry, safety, constraint, reward, `num_trailers`, and all perturbation parameters (`perturbation_level`, `perturbation_seed`, `control_noise_std`, `param_uncertainty_std`, `non_gaussian_dist`, `non_gaussian_df`, `state_disturbance_scale`, `disturbance_clip_factor`, `perturbed_l1`, `perturbed_l2`, `perturbed_lh`) from `config`.
3. Attaches `env.env = env_config` and updates `env.obs_circles` and `env.obs_rectangles`.
4. Applies `init_dx, init_dy, init_theta1, init_theta2` via `env.set_init_pos(...)` and `goal_theta1, goal_theta2` via `env.set_goal_pos(...)`.
5. Clears demonstration attributes and JIT cache (`clear_jit_cache()`) and returns `env`.

### `get_default_parking_config() -> Dict[str, Any]` (lines 168–184)
Returns a standard 2×8 parking lot configuration (`target_spaces=[3, 11]`).

### `create_tight_parking_config() -> Dict[str, Any]` (lines 187–203)
Returns a tighter 3×6 parking lot configuration (`space_width=3.0`, `space_length=6.5`, `target_spaces=[6, 12]`).
