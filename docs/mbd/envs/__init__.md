# `mbd/envs/__init__.py`

**Source file:** [mbd/envs/__init__.py](../../../mbd/envs/__init__.py)

## Overview
Environment factory module that instantiates the requested robot dynamics/environment wrapper given an `env_name` string, physical/planner parameters, and perturbation settings.

## Functions

### `get_env(...)` (lines 6–318)
Factory function returning a configured robot environment object.

- **Key Parameters**:
  - `env_name: str`: Target dynamics model (`"tt2d"`, `"n_trailer2d"`, `"acc_tt2d"`, or `"kinematic_bicycle2d"`).
  - `case: str = "navigation"`: Scenario type (`"parking"` or `"navigation"`).
  - `env_config=None`: Optional pre-configured `Env` object; if `None`, the robot class creates a default `Env(case=case)`.
  - `dt=0.2`, `H=50`: Integration timestep and planning horizon.
  - `motion_preference=0`: `0` (none), `1` (forward pref), `-1` (backward pref), `2` (strict forward), `-2` (strict backward).
  - Safety flags & penalties: `collision_penalty`, `hitch_penalty`, `enable_shielded_rollout_collision`, `enable_shielded_rollout_hitch`, `enable_projection`, `enable_guidance`.
  - Geometry & limits: `l1`, `l2`, `lh`, `lf1`, `lr`, `lf2`, `lr2`, `tractor_width`, `trailer_width`, `v_max`, `delta_max_deg`, `a_max`, `omega_max`, `num_trailers`.
  - Reward shaping & demo weights: `reward_threshold`, `ref_reward_threshold`, `max_w_theta`, `hitch_angle_weight`, `d_thr_factor`, `k_switch`, `steering_weight`, `preference_penalty_weight`, `heading_reward_weight`, `terminal_reward_threshold`, `terminal_reward_weight`, `ref_pos_weight`, `ref_theta1_weight`, `ref_theta2_weight`.
  - **Perturbation parameters (lines 47–57)**:
    - `perturbation_level: int = 0` (`0`: Nominal, `1`: Gaussian Control Noise, `2`: Gaussian Parameter Uncertainty, `3`: Non-Gaussian Disturbances)
    - `perturbation_seed: int | None = None` (`None` samples a fresh random seed per run)
    - `control_noise_std = (0.05, 0.05)`
    - `param_uncertainty_std: float = 0.10`
    - `non_gaussian_dist: str = "student_t"` (`"uniform"`, `"student_t"`, `"cauchy"`, `"laplace"`)
    - `non_gaussian_df: float = 2.5`
    - `state_disturbance_scale: float = 0.05`
    - `disturbance_clip_factor: float = 5.0`
    - `perturbed_l1: float | None = None`, `perturbed_l2: float | None = None`, `perturbed_lh: float | None = None`
  - **Robust tube tightening & Stage-2 iLQR parameters (lines 58–62)**:
    - `enable_robust_shielding: bool = False` (opt-in tube tightening for obstacle collision & hitch limits)
    - `tube_obstacle_margin = 0.0` (scalar float or 1D schedule array [m])
    - `tube_hitch_margin = 0.0` (scalar float or 1D schedule array [rad])
    - `enable_ilqr_backup: bool = False` (opt-in Stage-2 branchless JAX iLQR backup inside `AccTractorTrailer2d`)
    - `ilqr_iters: int = 4` (iterations for `solve_ilqr_backup`)
    - Forwarded via `robust_kwargs` to `TractorTrailer2d` (`env_name="tt2d"`, default `num_trailers=1`) and `AccTractorTrailer2d` (`env_name="acc_tt2d"`).
- **Dispatch Logic**:
  - `"tt2d"` with `num_trailers == 0` → `KinematicBicycle2d`
  - `"tt2d"` with `num_trailers >= 2` → `NTrailer2d`
  - `"tt2d"` (default `num_trailers == 1`) → `TractorTrailer2d`
  - `"n_trailer2d"` → `NTrailer2d`
  - `"acc_tt2d"` → `AccTractorTrailer2d`
  - `"kinematic_bicycle2d"` → `KinematicBicycle2d`
  - Any other string raises `ValueError(f"Unknown environment: {env_name}")`.

## Extension Points
- **New vehicle class**: Import the new class at the top of this file, add any new physical parameters to `get_env(...)`, and add an `elif env_name == "<your_env_name>":` branch forwarding `**perturbation_kwargs`.
