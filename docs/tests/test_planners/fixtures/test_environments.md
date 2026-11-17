# `tests/test_planners/fixtures/test_environments.py`

**Source file:** [tests/test_planners/fixtures/test_environments.py](../../../../tests/test_planners/fixtures/test_environments.py)

## Overview
Provides helper functions to build customized `Env` and robot environments from a `TestConfig` for planner tests.

## Functions

### `setup_parking_environment(config: TestConfig) -> Env` (lines 27–55)
Constructs `Env(case="parking", parking_config=config.custom_parking_config or None)` and appends `config.custom_circular_obstacles` (filtering out dummy zero-radius circles) and `config.custom_rectangular_obstacles`.

### `create_test_tt2d_environment(config: TestConfig) -> Any` (lines 58–153)
1. Calls `setup_parking_environment(config)` to create `env_config`.
2. Calls `mbd.envs.get_env(config.env_name, ...)` forwarding geometry, safety, constraint, and reward parameters from `config`.
3. Attaches `env.env = env_config`, updates `env.obs_circles`, `env.obs_rectangles`, and `env.obs`.
4. Applies `init_dx, init_dy, init_theta1, init_theta2` via `env.set_init_pos(...)` and `goal_theta1, goal_theta2` via `env.set_goal_pos(...)` if specified.
5. Resets `env.xref = None`, `env.rew_xref = None`, `env.angle_mask = None`, and calls `clear_jit_cache()`.
6. Returns `env`.

### `get_default_parking_config() -> Dict[str, Any]` (lines 156–172)
Returns a standard 2×8 parking lot configuration (`target_spaces=[3, 11]`).

### `create_tight_parking_config() -> Dict[str, Any]` (lines 175–191)
Returns a tighter 3×6 parking lot configuration (`space_width=3.0`, `space_length=6.5`, `target_spaces=[6, 12]`).

## Extension Notes
- **New vehicle class / Perturbations**: Notice `create_test_tt2d_environment` explicitly enumerates arguments passed to `mbd.envs.get_env(...)` (lines 73–111) and does **not** currently pass `enable_projection`, `enable_guidance`, or `num_trailers` (or any custom parameters you add to `MBDConfig`). Forward any new vehicle or perturbation parameters here so tests can configure them. Also note lines 113–121 overwrite `env.env` and `env.obs_*` after `get_env(...)`—passing `env_config=env_config` directly to `get_env` ensures `__init__` sees custom obstacles too.
