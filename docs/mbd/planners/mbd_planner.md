# `mbd/planners/mbd_planner.py`

**Source file:** [mbd/planners/mbd_planner.py](../../../mbd/planners/mbd_planner.py)

## Overview
Implements the core **Safe Model Predictive Diffusion (Safe-MPD)** planner: configuration (`MBDConfig`), JIT function caching, reverse diffusion sampling (`run_diffusion` / `reverse_once`), steering smoothness reward, post-processing nominal and perturbed trajectory rollouts, and result saving/rendering.

---

## Global Utilities & Cache

### `count_compilation(f)` (lines 30–37)
Decorator incrementing global `compilation_count` and logging debug messages whenever a decorated function is compiled/invoked.

### `clear_jit_cache()` (lines 43–49)
Clears `_jit_function_cache` (`dict`). Called automatically by `env.set_perturbation(...)` and between test runs when static parameters or environment attributes change.

---

## Configuration

### `@dataclass class MBDConfig` (lines 54–139)
Stores all planner, safety, vehicle geometry, reward, animation, scalability, perturbation, and robust shielding parameters:
- **Experiment/Env**: `seed`, `env_name` (`"acc_tt2d"`, `"tt2d"`, `"n_trailer2d"`, `"kinematic_bicycle2d"`), `case` (`"parking"` or `"navigation"`), `verbose`, `dt`, `num_trailers`.
- **Diffusion**: `Nsample` (20000), `Hsample` (50), `Ndiffuse` (100), `temp_sample` (1e-5), `beta0` (1e-5), `betaT` (1e-2), `enable_demo`.
- **Motion & Safety**: `motion_preference` (`0`, `1`, `-1`, `2`, `-2`), `collision_penalty`, `hitch_penalty`, `enable_shielded_rollout_collision`, `enable_shielded_rollout_hitch`, `enable_projection`, `enable_guidance`.
- **Physical Geometry & Limits**: `l1`, `l2`, `lh`, `lf1`, `lr`, `lf2`, `lr2`, `tractor_width`, `trailer_width`, `v_max`, `delta_max_deg`, `a_max`, `omega_max`.
- **Reward Shaping**: `reward_threshold`, `ref_reward_threshold`, `max_w_theta`, `hitch_angle_weight`, `terminal_reward_threshold`, `terminal_reward_weight`, `d_thr_factor`, `k_switch`, `steering_weight`, `preference_penalty_weight`, `heading_reward_weight`, `ref_pos_weight`, `ref_theta1_weight`, `ref_theta2_weight`.
- **Rendering**: `render`, `save_animation`, `show_animation`, `save_denoising_animation`, `frame_skip`.
- **Perturbations (lines 121–132)**:
  - `perturbation_level: int = 0` (`0`: Nominal, `1`: Gaussian Control Noise, `2`: Gaussian Parameter Uncertainty, `3`: Non-Gaussian Disturbances)
  - `perturbation_seed: int | None = None`
  - `control_noise_std: tuple = (0.05, 0.05)`
  - `param_uncertainty_std: float = 0.10`
  - `non_gaussian_dist: str = "student_t"`
  - `non_gaussian_df: float = 2.5`
  - `state_disturbance_scale: float = 0.05`
  - `disturbance_clip_factor: float = 5.0`
  - `perturbed_l1: float | None = None`, `perturbed_l2: float | None = None`, `perturbed_lh: float | None = None`
- **Robust Tube Tightening & Stage-2 iLQR Backup (lines 133–138)**:
  - `enable_robust_shielding: bool = False`
  - `tube_obstacle_margin: float | tuple = 0.0`
  - `tube_hitch_margin: float | tuple = 0.0`
  - `enable_ilqr_backup: bool = False`
  - `ilqr_iters: int = 4`

### `_coerce_margin_config(val)` & `dict_to_config_obj(config_dict)` (lines 141–231)
Normalizes scalar or sequence tube margins (`tuple(float(x) for x in arr)`) and converts a Python `dict` into an `MBDConfig` dataclass, including perturbation and robust shielding fields.

---

## Core Planner

### `run_diffusion(args=None, env=None)` (lines 215–840)
Runs the full Safe-MPD planning loop on `env` using `args` (`MBDConfig` or `dict`).
- **Returns**: `(rew_final, Y0, trajectory_states, timing_info)`
  - `rew_final`: Scalar total reward of final `Y0` on the nominal model.
  - `Y0`: Final denoised (and shielded/applied) control sequence, shape `(Hsample, Nu)`.
  - `trajectory_states`: Nominal rolled-out state sequence, shape `(Hsample + 1, Nx)`.
  - `timing_info`: Breakdown dict (`total_time`, `demo_generation_time`, `compilation_time`, `jit_setup_time`, `warmup_time`, `reward_compile_time`, `diffusion_setup_time`, `pure_diffusion_time`, `post_processing_time`, `overhead_time`, `perturbation_info`).
  - Also populates `env.nominal_trajectory_states`, `env.perturbed_trajectory_states`, and `env.perturbed_actions`.

#### Execution Stages & Nested Functions inside `run_diffusion`
1. **Perturbation & Robust Margin Sync & Demo Setup**: Syncs perturbation settings and robust margins (`env.set_robust_margins(...)`) from `args` to `env` unless manually overridden; generates demonstration trajectory if `args.enable_demo` is True.
2. **JIT Setup & Warmup**: Invalidates `_jit_function_cache` whenever the `env` instance or robust configuration signature (`id(env)`, `enable_robust_shielding`, `enable_ilqr_backup`, `ilqr_iters`, step-0 margins) changes, then compiles/caches `step_env_jit`, `rollout_us_jit`, and `rollout_us_with_terminal_jit`.
3. **Diffusion Schedule & `compute_steering_reward(Y0s)`**: Computes linear $\beta_t$ schedule and trajectory-level steering magnitude + rate reward.
4. **`reverse_once(carry, state_init, config_params_tuple)` & `reverse(YN, rng, state_init)`**:
   - Runs $N_{\text{diffuse}}-1$ reverse denoising iterations using `jax.vmap(rollout_us_with_terminal_jit)` on the nominal model.
   - When `enable_robust_shielding` or `enable_ilqr_backup` is active, zeroes out Monte Carlo sample weights for candidate trajectories already unsafe/unrecoverable at $t = 0$ (falling back cleanly to softmax over `logp0` if zero candidates survive).
   - Replaces final `Y0` with `Y0_applied` from `rollout_us_with_terminal_jit(state_init, Y0)`.
5. **Post-Processing Nominal & Perturbed Rollouts (lines 562–670)**:
   - Rolls out `Y0` on the nominal model (`step_env_jit_final_rollout`) to obtain `nominal_trajectory_states` (`trajectory_states`).
   - Rolls out `Y0` on the perturbed simulation environment via `rollout_perturbed_us(jax.jit(env.step_perturbed), state_pert_init, Y0)` to obtain `perturbed_trajectory_states` and `perturbed_actions`.
   - Stores `env.nominal_trajectory_states`, `env.perturbed_trajectory_states`, and `env.perturbed_actions`.
6. **Saving & Rendering (lines 693–820)**:
   - Always saves `mu_0ts.npy`, `final_actions.npy`, `perturbed_actions.npy`, `nominal_trajectory_states.npy`, `perturbed_trajectory_states.npy`, `trajectory_states.npy`, and `env_metadata.json` under `results/{args.env_name}/`.
   - When `args.render` is True: plots `nominal_trajectory_states` and overlays `perturbed_trajectory_states` (when `perturbation_level > 0`) in `rollout.png` and `rollout.svg`.
   - When `args.save_animation` or `args.show_animation` is True: animates the nominal trajectory (`nominal_trajectory_animation.mp4`) and then, when `perturbation_level > 0`, animates the perturbed trajectory (`perturbed_trajectory_animation.mp4`) with the nominal trajectory overlaid.

---

## Standalone Entry Point (`__main__`, lines 845–933)
Creates default `MBDConfig()`, builds `env` via `mbd.envs.get_env(...)` forwarding all physical and perturbation parameters, sets initial/goal poses, and calls `run_diffusion(args=config, env=env)`.
