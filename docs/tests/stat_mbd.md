# `tests/stat_mbd.py`

**Source file:** [tests/stat_mbd.py](../../tests/stat_mbd.py)

## Overview
Batch Monte Carlo statistical evaluator for Safe-MPD. Samples valid initial conditions, runs `run_diffusion` across `num_trials`, evaluates both nominal and perturbed trajectories, computes aggregate statistics, and optionally generates spatial success/failure heat maps (`heat_map.png`/`.svg`) and merged progress videos.

---

## Dataclasses

### `@dataclass class StatisticalResults` (lines 40–50)
Holds aggregated metrics: `success_rate`, `avg_position_error`, `collision_rate`, `jackknife_rate`, `avg_pure_diffusion_time`, `std_pure_diffusion_time`, `num_trials`, `successful_trials`, `individual_results`.

### `@dataclass class TrialConfig` (lines 54–60)
Represents one sampled initial condition: `dx`, `dy`, `theta1`, `theta2`, `trial_id`.

---

## Functions

- **`sample_initial_conditions(num_trials: int, seed: int = 42) -> List[TrialConfig]`** (lines 63–108): Samples valid collision- and jackknife-free initial poses.
- **`get_default_parking_config()`** (lines 111–126) & **`create_rectangular_obstacles()`** (lines 128–141): Local parking obstacle setup.
- **`is_valid_initial_condition(trial_config: TrialConfig) -> bool`** (lines 144–202): Checks candidate initial condition against obstacles and hitch limits.
- **`create_trial_environment(base_config: MBDConfig, trial_config: TrialConfig)`** (lines 205–297): Constructs the trial environment via `mbd.envs.get_env(...)`, forwarding all physical, safety, reward, and perturbation parameters (using `(123 if base_config.perturbation_seed is None else base_config.perturbation_seed) + trial_config.trial_id` as the per-trial perturbation seed).
- **`evaluate_trial_result(trajectory_states: jnp.ndarray, env, goal_position_threshold: float = 4.0) -> Dict`** (lines 300–359): Evaluates final goal distance error, trajectory obstacle collisions, and hitch angle violations.
- **`run_statistical_evaluation(...) -> StatisticalResults`** (lines 362–521): Runs all trials, evaluates `env.perturbed_trajectory_states` when `perturbation_level > 0` (and records both `nominal_result` and `perturbed_result` per trial), aggregates metrics, and optionally saves heat maps / progress animations.
- **`create_heat_map_visualization(...)`** (lines 524–618): Plots initial trial positions color-coded by success/failure.
- **`main()`** (lines 622–673): Runs 100 parking trials and displays the heat map.
