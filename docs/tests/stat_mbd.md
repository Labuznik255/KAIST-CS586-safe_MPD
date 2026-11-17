# `tests/stat_mbd.py`

**Source file:** [tests/stat_mbd.py](../../tests/stat_mbd.py)

## Overview
Batch Monte Carlo statistical evaluator for Safe-MPD. Samples valid initial conditions, runs `run_diffusion` across `num_trials`, checks trajectory safety and goal accuracy, computes aggregate statistics, and optionally generates spatial success/failure heat maps (`heat_map.png`/`.svg`) and merged progress videos.

---

## Dataclasses

### `@dataclass class StatisticalResults` (lines 39–50)
Holds aggregated metrics: `success_rate`, `avg_position_error`, `collision_rate`, `jackknife_rate`, `avg_pure_diffusion_time`, `std_pure_diffusion_time`, `num_trials`, `successful_trials`, `individual_results`.

### `@dataclass class TrialConfig` (lines 53–60)
Represents one sampled initial condition: `dx`, `dy`, `theta1`, `theta2`, `trial_id`.

---

## Functions

### `sample_initial_conditions(num_trials: int, seed: int = 42) -> List[TrialConfig]` (lines 63–108)
Uses `np.random.RandomState(seed)` to sample `dx ~ U(-12, 12)`, `dy ~ U(2, 7)`, `theta1` near `0` or `pi` (`±pi/10`), and `theta2 = theta1 + U(-pi/10, pi/10)`. Filters candidates with `is_valid_initial_condition` (up to `10 * num_trials` attempts).

### `get_default_parking_config()` (lines 111–126)
Returns a local parking dictionary (`occupied_spaces=[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]`, `target_spaces=[4, 12]`, `obstacle_radius=1.0`) copied/customized from `Env.get_default_parking_config`.

### `create_rectangular_obstacles()` (lines 128–141)
Returns `[[0.0, -14.0, 30.0, 1.0, 0.0]]` (bottom parking boundary wall).

### `is_valid_initial_condition(trial_config: TrialConfig) -> bool` (lines 144–202)
Builds a temporary `tt2d` parking environment with local obstacles, applies `trial_config` via `set_init_pos`, and returns `not (obstacle_collision or hitch_violation)`.

### `create_trial_environment(base_config: MBDConfig, trial_config: TrialConfig)` (lines 205–286)
Constructs the environment for one trial by forwarding all `base_config` fields to `mbd.envs.get_env(...)`, overriding `env.obs_rectangles` and `env.env.obs_rectangles` with `create_rectangular_obstacles()`, setting initial pose from `trial_config`, and setting goal orientation `(-pi/2, -pi/2)`.

### `evaluate_trial_result(trajectory_states: jnp.ndarray, env, goal_position_threshold: float = 4.0) -> Dict` (lines 289–348)
Evaluates a single completed trajectory:
- Computes `tractor_pos_error` (using `final_state_4d[:2]` vs `env.xg[:2]`) and `trailer_pos_error` (using `env.get_trailer_back_position(final_state_4d)` vs `env.xg[:2]`), taking `final_position_error = min(tractor_pos_error, trailer_pos_error)`.
- Loops over all timesteps `t`, slicing `state_4d = trajectory_states[t][:4]`, checking `env.check_obstacle_collision` and `env.check_hitch_violation`.
- Marks `success = (final_position_error <= goal_position_threshold) and not obstacle_collision and not hitch_violation`.

### `run_statistical_evaluation(config: MBDConfig, num_trials: int = 20, seed: int = 42, verbose: bool = True, show_heat_map: bool = False, save_progress_animation: bool = False) -> StatisticalResults` (lines 351–507)
Runs `sample_initial_conditions`, iterates through each `trial_config`, creates the environment, runs `run_diffusion`, calls `evaluate_trial_result`, optionally renders per-trial progress animations with cumulative success/fail markers (`create_animation` + `merge_progress_videos`), aggregates statistics into `StatisticalResults`, and optionally generates a heat map.

### `create_heat_map_visualization(individual_results: List[Dict], env, verbose: bool = True, return_fig: bool = False)` (lines 510–604)
Plots the environment (`env.render(ax, empty)`) and overlays green circles (success) / red crosses (fail) at each trial's world `(x0[0], x0[1])`. Either returns the Matplotlib `Figure` (for W&B logging in `tune_mbd.py`) or saves `heat_map.png`/`.svg` and displays it.

### `main()` (lines 608–660)
Configures `"tt2d"` parking parameters and runs 100 trials (`show_heat_map=True`).

---

## Extension Notes
- **New vehicle class**:
  - Forward any new `MBDConfig` fields inside `create_trial_environment` (`mbd.envs.get_env(...)`, lines 221–265).
  - Note `evaluate_trial_result` (lines 305, 326) slices `[:4]`! If your new vehicle has `num_trailers >= 2` or uses full state for collision/hitch checks, pass the full state when appropriate; if it is a single-body vehicle without a trailer (`KinematicBicycle2d`), `get_trailer_back_position` fallback must exist.
  - Notice `is_valid_initial_condition` (line 162) hardcodes `env_name="tt2d"`.
- **Perturbations**:
  - `stat_mbd.py` is the ideal place to run **Monte Carlo robustness benchmarks** of perturbed tractor-trailer models (either by passing perturbation parameters through `base_config` -> `create_trial_environment`, or by rolling out `Y0` through a perturbed simulator before calling `evaluate_trial_result`).
