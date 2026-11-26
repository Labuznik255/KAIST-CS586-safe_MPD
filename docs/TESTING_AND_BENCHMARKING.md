# Testing, Benchmarking, and Parameter Reference Guide

This guide explains **how to run every test, calibration, benchmark, and visualization script** in Safe-MPD, and documents **all configurable parameters and where to set them**.

---

## 1. Environment Setup & Activation

To replicate the tested environment (`Python >=3.12, <3.15`; tested on `3.14.8`):

```bash
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e . --no-deps
```

Run commands from the repository root using the project virtual environment:

```bash
./.venv/bin/python <script> [args]
```

### GPU / Virtualized CUDA (GRID A100D) & CPU Execution Flags
- **Virtualized NVIDIA GRID GPUs (e.g., A100D-20C)**: JAX/XLA may fail with `no supported devices found for platform CUDA ... VMM is required`. Disable XLA's experimental VMM allocation:
  ```bash
  XLA_FLAGS="--xla_gpu_experimental_vmm_disabled" ./.venv/bin/python tests/test_robust_rmps.py --Nsample 20000 --Ndiffuse 100
  ```
- **CPU-only / fallback execution**: Force CPU platform cleanly (suppresses CUDA plugin init warnings):
  ```bash
  JAX_PLATFORMS=cpu ./.venv/bin/python tests/test_robust_rmps.py --seeds 5 --Nsample 800 --Ndiffuse 25
  ```
- **Headless plotting / CI**: Prepend `MPLBACKEND=Agg` whenever running without a display server.

---

## 2. How to Run Tests and Benchmarks

### A. Unit & Scenario Test Suite (`tests/test_planners/run_tests.py`)
Runs deterministic `unittest` scenario tests across kinematic (`tt2d`) and acceleration-controlled (`acc_tt2d`) parking/navigation tasks ([docs/tests/test_planners/run_tests.md](tests/test_planners/run_tests.md)).

```bash
# Default: run kinematic scenario tests
./.venv/bin/python tests/test_planners/run_tests.py

# List all test methods by category
./.venv/bin/python tests/test_planners/run_tests.py --list

# Run acceleration (acc_tt2d), demo, or full suite
./.venv/bin/python tests/test_planners/run_tests.py --acc
./.venv/bin/python tests/test_planners/run_tests.py --demo
./.venv/bin/python tests/test_planners/run_tests.py --all --timeout 180

# Run a single test with GUI visualization enabled
./.venv/bin/python tests/test_planners/run_tests.py --single test_parking_basic_forward --visualize
```

### B. Single Navigation Planning + Perturbed Rollout (`tests/test_mbd.py`)
Runs Safe-MPD on a 32×32m navigation obstacle course, executes the planned actions on both the nominal model and perturbed `env.step_perturbed`, and saves all artifacts under `results/{env_name}/` ([docs/tests/test_mbd.md](tests/test_mbd.md)).

```bash
./.venv/bin/python tests/test_mbd.py
```
Edit `MBDConfig(...)` and `mbd.envs.get_env(...)` (or uncomment `env.set_perturbation(...)`) inside [tests/test_mbd.py](../tests/test_mbd.py) to switch vehicle models (`tt2d`, `acc_tt2d`, `kinematic_bicycle2d`, `n_trailer2d`), diffusion budgets (`Nsample`, `Ndiffuse`), or perturbation levels (0–3).

### C. Perturb an Already-Planned Trajectory (`tests/test_perturbations.py`)
Re-rolls out an existing `final_actions.npy` under any perturbation level without re-running diffusion ([docs/tests/test_perturbations.md](tests/test_perturbations.md)).

```bash
# Level 1 Gaussian control noise
./.venv/bin/python tests/test_perturbations.py results/acc_tt2d --level 1 --control-noise-std 0.08 0.05 --seed 42

# Level 2 time-invariant link-length uncertainty
./.venv/bin/python tests/test_perturbations.py results/acc_tt2d --level 2 --param-uncertainty-std 0.15 --seed 123

# Level 2 explicit perturbed lengths (no random draw)
./.venv/bin/python tests/test_perturbations.py results/acc_tt2d --level 2 --perturbed-l1 3.6 --perturbed-l2 3.2 --perturbed-lh 1.3

# Level 3 heavy-tailed non-Gaussian disturbances (headless)
./.venv/bin/python tests/test_perturbations.py results/acc_tt2d --level 3 --dist student_t --df 2.5 --state-disturbance-scale 0.05 --no-visualize
```

### D. Stage 1: Offline Tube Calibration (`mbd/robust/tube_calibration.py`)
Precomputes empirical disturbance tubes $B(t)$ and constraint-tightening margins (`r_tube`, `b_hitch`, `b_v`) for `AccTractorTrailer2d` across Levels 0–3 ([docs/mbd/robust/tube_calibration.md](mbd/robust/tube_calibration.md)).

```bash
# Default: 1000 Monte Carlo rollouts per level, T=25, quantile=0.99
./.venv/bin/python mbd/robust/tube_calibration.py

# High-precision worst-case calibration (quantile=1.0, 2000 rollouts)
./.venv/bin/python mbd/robust/tube_calibration.py --N-mc 2000 --horizon 25 --dt 0.25 --quantile 1.0 --seed 42 --out-dir results/tubes
```

### E. Stage 5: Comparative RMPS Benchmark (`tests/test_robust_rmps.py`)
Compares `baseline`, `tube_only`, `rmps_no_replan`, and `rmps_replan` across Perturbation Levels 0–3 ([docs/tests/test_robust_rmps.md](tests/test_robust_rmps.md)).

```bash
# Fast CPU benchmark (default settings, all 4 variants)
JAX_PLATFORMS=cpu MPLBACKEND=Agg ./.venv/bin/python tests/test_robust_rmps.py --seeds 5 --Nsample 800 --Ndiffuse 25

# Compare baseline vs. rmps_replan on explicit seeds (including seed 9123)
JAX_PLATFORMS=cpu MPLBACKEND=Agg ./.venv/bin/python tests/test_robust_rmps.py --variants baseline rmps_replan --levels 0 1 2 3 --seed-list 9123 42 43 44 45 --Nsample 800 --Ndiffuse 25 --max-steps 150 --max-replans 5 --out-dir results/rmps_benchmark_fixed

# Run only the NMPC + replanning variant on Level 3
JAX_PLATFORMS=cpu MPLBACKEND=Agg ./.venv/bin/python tests/test_robust_rmps.py --variants rmps_replan --levels 3 --seeds 5 --out-dir results/rmps_replan_only

# Full GPU benchmark on NVIDIA GRID A100D-20C (with VMM disabled)
XLA_FLAGS="--xla_gpu_experimental_vmm_disabled" MPLBACKEND=Agg ./.venv/bin/python tests/test_robust_rmps.py --seeds 20 --levels 0 1 2 3 --Nsample 20000 --Ndiffuse 100 --max-steps 150 --max-replans 5 --recalibrate --out-dir results/rmps_gpu
```
Outputs:
- `summary.json` and `summary.csv` (success rate, collision rate, hitch violation rate, CVaR error & clearance, intervention rate, replans, runtime/step, and failure type counts: collision, jackknife, collision+jackknife, safe timeout, safe max-replans, safe stopped)
- `benchmark_comparison.png` (4-panel grouped bar plot)
- `all_trajectories_overview.png` (level × variant grid overlaying all seeds' trajectories color-coded by outcome)
- `episodes/level{L}/{variant}/seed_{S}/` (when `--save-episodes all` [default] or `per_level`, containing `.npy` trajectories, `env_metadata.json`, and `rmps_diagnostics.png` ready for `render.py`)

### F. Standalone IPOPT NMPC Planner & Benchmark (`tests/test_nmpc.py`)
Tests CasADi + IPOPT NMPC on the exact same scenarios and perturbation levels as Safe-MPD ([tests/test_nmpc.py](../tests/test_nmpc.py)).

```bash
# Default navigation task on acc_tt2d (full-horizon optimal control)
./.venv/bin/python tests/test_nmpc.py --env acc_tt2d --case navigation --level 1

# Receding-horizon closed-loop MPC
./.venv/bin/python tests/test_nmpc.py --env acc_tt2d --case navigation --mpc-mode --level 3

# Verify all 4 vehicle types headless
./.venv/bin/python tests/test_nmpc.py --all-vehicles --no-render
```

### G. Pure Diffusion Timing Benchmark (`mbd/scripts/benchmark_diffusion.py`)
Measures pure diffusion runtime separately from JAX JIT compilation across `Small`, `Medium`, and `Large` sample budgets ([docs/mbd/scripts/benchmark_diffusion.md](mbd/scripts/benchmark_diffusion.md)).

```bash
./.venv/bin/python mbd/scripts/benchmark_diffusion.py
```

### H. Monte Carlo Parking Evaluation & Tuning (`tests/stat_mbd.py`, `tests/tune_mbd.py`)
- **Batch spatial evaluation (`tests/stat_mbd.py`)**: Samples 100 random collision-free parking initial poses, runs `run_diffusion` across nominal and perturbed models, and plots `heat_map.png`/`.svg` ([docs/tests/stat_mbd.md](tests/stat_mbd.md)).
  ```bash
  ./.venv/bin/python tests/stat_mbd.py
  ```
- **Optuna + W&B tuning (`tests/tune_mbd.py`)**: Optimizes reward/diffusion hyperparameters using GP sampling and `MedianPruner` ([docs/tests/tune_mbd.md](tests/tune_mbd.md)).
  ```bash
  ./.venv/bin/python tests/tune_mbd.py
  ```

### I. Interactive & Video Trajectory Visualizer (`render.py`)
Visualizes any results directory (`results/acc_tt2d`, `results/rmps_benchmark/sample_rmps_replan_level3`, etc.) or `.npz` test archive ([docs/render.md](render.md)).

```bash
# Interactive animation only (skip static plot)
./.venv/bin/python render.py results/rmps_benchmark/sample_rmps_replan_level3 --mode both --no-static

# Show perturbed/RMPS closed-loop animation only
./.venv/bin/python render.py results/rmps_benchmark/sample_rmps_replan_level3 --mode perturbed --no-static

# Static plot only (saves rollout.png & rollout.svg)
./.venv/bin/python render.py results/rmps_benchmark/sample_rmps_replan_level3 --mode both --no-animate

# Export MP4 videos
./.venv/bin/python render.py results/rmps_benchmark/sample_rmps_replan_level3 --mode both --no-static --save-animation
```

---

## 3. Complete Parameter Reference & Where to Set Them

### 3.1 Diffusion & Core Experiment (`MBDConfig` in [mbd/planners/mbd_planner.py](../mbd/planners/mbd_planner.py#L54-L139))
Set on `MBDConfig(...)` (or passed as a `dict` converted via `dict_to_config_obj`) and passed to `run_diffusion(args=config, env=env)` or `SafeMPDPlanner(config, env)`:

| Parameter | Default | Meaning |
|---|---|---|
| `seed` | `42` | JAX PRNG seed for diffusion sampling |
| `env_name` | `"acc_tt2d"` | Vehicle model: `"acc_tt2d"`, `"tt2d"`, `"kinematic_bicycle2d"`, `"n_trailer2d"` |
| `case` | `"parking"` | Scenario preset: `"parking"` or `"navigation"` (custom obstacles) |
| `verbose` | `False` | Verbose logging flag |
| `dt` | `0.25` | Simulation timestep $\Delta t$ [s] (also passed to `get_env(..., dt=...)`; `get_env` default is `0.2`) |
| `Nsample` | `20000` | Number of parallel Monte Carlo control trajectories per diffusion step |
| `Hsample` | `50` | Planning horizon steps $H$ (also pass `H=config.Hsample` to `get_env`) |
| `Ndiffuse` | `100` | Number of reverse denoising iterations |
| `temp_sample` | `1e-5` | Softmax inverse temperature controlling concentration on high-reward samples |
| `beta0` / `betaT` | `1e-5` / `1e-2` | Linear diffusion variance schedule endpoints |
| `enable_demo` | `False` | Enable waypoint demonstration guidance |
| `num_trailers` | `1` | Trailer count (`0` = bicycle, `1` = single trailer, `>=2` = N-trailer) |

### 3.2 Physical Vehicle Geometry & Input Limits
Set in **both** `MBDConfig(...)` (for metadata/standalone `__main__`) **and** `mbd.envs.get_env(...)` ([mbd/envs/__init__.py](../mbd/envs/__init__.py#L6-L63)) or robot `__init__` constructors:

| Parameter | Default | Meaning |
|---|---|---|
| `l1`, `l2`, `lh` | `3.23`, `2.9`, `1.15` | Nominal tractor wheelbase, trailer axle-to-hitch length, hitch overhang [m] |
| `lf1`, `lr` | `0.9`, `1.848` | Tractor front overhang & rear overhang [m] |
| `lf2`, `lr2` | `1.42`, `3.225` | Trailer front & rear overhang [m] |
| `tractor_width`, `trailer_width` | `2.0`, `2.5` | Vehicle body widths [m] |
| `v_max` | `3.0` | Speed limit [m/s] |
| `delta_max_deg` | `55.0` | Steering angle limit [deg] |
| `a_max` | `2.0` | Acceleration limit [m/s²] (`AccTractorTrailer2d`) |
| `omega_max` | `1.0` | Steering rate limit [rad/s] (`AccTractorTrailer2d`) |
| `trailer_lengths` | `[2.9, ...]` | Optional per-trailer lengths (`NTrailer2d` constructor in [mbd/robots/n_trailer2d.py](../mbd/robots/n_trailer2d.py)) |
| `hitch_lengths` | `[1.15, ...]` | Optional per-hitch offsets (`NTrailer2d` constructor) |

### 3.3 Safety Enforcement & Reward Shaping (`MBDConfig` & `mbd.envs.get_env`)

| Parameter | `MBDConfig` Default | `get_env` Default | Meaning |
|---|---|---|---|
| `motion_preference` | `0` | `0` | `0` none, `1`/`-1` forward/backward pref, `2`/`-2` strict |
| `collision_penalty` | `0.15` | `0.15` | Naive stage penalty on obstacle collision |
| `hitch_penalty` | `0.10` | `0.10` | Naive stage penalty on hitch-angle violation |
| `enable_shielded_rollout_collision` | `True` | `True` | Enable rollout safety shield for obstacle collisions |
| `enable_shielded_rollout_hitch` | `True` | `True` | Enable rollout safety shield for jackknife/hitch limits |
| `enable_projection` | `False` | `False` | SLSQP control projection onto safe set |
| `enable_guidance` | `False` | `False` | Gradient-descent safety guidance |
| `reward_threshold` | `50.0` | `25.0` | Position error scale for stage reward |
| `ref_reward_threshold` | `10.0` | `5.0` | Position error scale for demo evaluation |
| `max_w_theta` | `0.5` | `0.75` | Max heading weight in stage reward |
| `hitch_angle_weight` | `0.01` | `0.2` | Hitch articulation regularization weight |
| `terminal_reward_threshold` | `20.0` | `10.0` | Position error scale for terminal reward |
| `terminal_reward_weight` | `5.0` | `1.0` | Weight on terminal reward term |
| `d_thr_factor`, `k_switch` | `1.0`, `2.5` | `1.0`, `2.5` | Distance threshold multiplier & logistic heading switch slope |
| `steering_weight` | `0.01` | `0.05` | Trajectory-level steering smoothness cost weight |
| `preference_penalty_weight` | `0.5` | `0.05` | Penalty weight on motion preference violations |
| `heading_reward_weight` | `0.5` | `0.5` | Weight blending tractor vs. trailer heading reward |
| `ref_pos_weight`, `ref_theta1_weight`, `ref_theta2_weight` | `0.8`, `0.1`, `0.1` | `0.3`, `0.5`, `0.2` | Weights in demonstration log-probability reward |

### 3.4 Perturbation Framework (Levels 0–3)
Can be set in **four places** (highest runtime priority last):
1. `MBDConfig(...)` / `NMPCConfig(...)` (auto-synced inside `run_diffusion` / `run_nmpc` unless overridden on `env`).
2. `mbd.envs.get_env(...)` ([mbd/envs/__init__.py](../mbd/envs/__init__.py#L47-L57)).
3. `env.set_perturbation(level=..., seed=..., ...)` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py#L382-L419)) — overrides `MBDConfig` values when `run_diffusion` runs.
4. CLI flags in [tests/test_perturbations.py](../tests/test_perturbations.py#L214-L223) and [tests/test_nmpc.py](../tests/test_nmpc.py#L190-L191).

| Parameter | Default | Active Levels | Meaning |
|---|---|---|---|
| `perturbation_level` | `0` | 0, 1, 2, 3 | `0` nominal, `1` Gaussian control, `2` param uncertainty, `3` non-Gaussian |
| `perturbation_seed` | `None` | 0, 1, 2, 3 | Fixed `int` seed or `None` for fresh random seed per run |
| `control_noise_std` | `(0.05, 0.05)` | 1, 3 | Std/scale added in normalized `[-1, 1]^2` control space |
| `param_uncertainty_std` | `0.10` (`0.15` in `test_perturbations.py`) | 2 | Relative std for time-invariant link-length uncertainty |
| `non_gaussian_dist` | `"student_t"` | 3 | `"uniform"`, `"student_t"`, `"cauchy"`, or `"laplace"` |
| `non_gaussian_df` | `2.5` | 3 | Degrees of freedom for `"student_t"` |
| `state_disturbance_scale` | `0.05` | 3 | Additive process noise scale on $\dot{x}$ |
| `disturbance_clip_factor` | `5.0` | 3 | Clipping factor in units of scale for heavy-tailed draws |
| `perturbed_l1`, `perturbed_l2`, `perturbed_lh` | `None` | 2 | Optional explicit length overrides [m] bypassing random sampling |

### 3.5 Tube Tightening & Stage-2 Vectorized iLQR Backup (Planning Layer)
Set via `MBDConfig(...)`, `mbd.envs.get_env(...)`, or `env.set_robust_margins(r_tube=..., b_hitch=..., enable_robust_shielding=..., enable_ilqr_backup=..., ilqr_iters=...)` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py#L480-L504)):

| Parameter | Default | Where Set | Meaning |
|---|---|---|---|
| `enable_robust_shielding` | `False` | `MBDConfig`, `get_env`, `set_robust_margins` | Enables tube tightening & $t=0$ unsafe weight masking |
| `tube_obstacle_margin` (`r_tube`) | `0.0` | `MBDConfig`, `get_env`, `set_robust_margins` | Scalar or 1D horizon schedule [m] inflating obstacle circles & rectangles |
| `tube_hitch_margin` (`b_hitch`) | `0.0` | `MBDConfig`, `get_env`, `set_robust_margins` | Scalar or 1D horizon schedule [rad] tightening `phi_max` |
| `enable_ilqr_backup` | `False` | `MBDConfig`, `get_env`, `set_robust_margins` | Enables Stage-2 branchless JAX iLQR fallback inside `AccTractorTrailer2d.shielded_rollout` |
| `ilqr_iters` | `4` | `MBDConfig`, `get_env`, `set_robust_margins` | Number of `jax.lax.scan` iLQR iterations |
| `w_u`, `w_v_stage`, `w_obs`, `w_hitch`, `w_v_term`, `w_hitch_term`, `lm_reg` | `0.25`, `0.4`, `120.0`, `100.0`, `200.0`, `15.0`, `1e-2` | `ILQRBackupParams` ([mbd/robust/ilqr_backup.py](../mbd/robust/ilqr_backup.py#L21-L46)) | Smooth iLQR backup cost weights and Levenberg-Marquardt damping |
| `N_mc`, `T_horizon`, `quantile`, `seed` | `1000`, `25`, `0.99`, `42` | `calibrate_tubes(...)` or CLI | Offline Monte Carlo tube calibration parameters |

### 3.6 Online CPU NMPC Shielding & Replanning (Execution Layer)
Set on `RobustNMPCBackup(...)` ([mbd/robust/nmpc_backup.py](../mbd/robust/nmpc_backup.py#L51-L60)) and `run_rmps_episode(...)` ([mbd/robust/rmps_runner.py](../mbd/robust/rmps_runner.py#L133-L150)):

| Parameter | Default | Where Set | Meaning |
|---|---|---|---|
| `T_bak` | `12` | `RobustNMPCBackup` | Backup stopping & tracking NMPC horizon steps |
| `v_eps` | `0.05` | `RobustNMPCBackup` | Equilibrium stopped-velocity threshold $\|v\| \le v_{\varepsilon}$ [m/s] |
| `max_iter` | `40` | `RobustNMPCBackup` | IPOPT max iterations for online recovery & tracking solves |
| `disc_safety_pad` | `0.01` | `RobustNMPCBackup` | Extra obstacle boundary buffer [m] inside CasADi constraints |
| `hitch_safety_pad` | `0.01` | `RobustNMPCBackup` | Extra hitch angle buffer [rad] inside recovery NLP |
| `cache_size` | `128` | `RobustNMPCBackup` | LRU cache entries shared between `is_recoverable` & `initialize_backup` |
| `max_steps`, `max_replans` | `150` (`120` in benchmark), `5` (`2` in benchmark) | `run_rmps_episode`, `test_robust_rmps.py` | Episode horizon cap & online Safe-MPD replan cap |
| `enable_nmpc_shield` | `True` | `run_rmps_episode` | Toggles online RMPS recoverability monitor & NMPC shield |
| `enable_replanning` | `True` | `run_rmps_episode` | Toggles Safe-MPD replanning once stopped inside $G_e$ |
| `use_delay_compensation` | `True` | `run_rmps_episode` | Predicts `f_nom(x, u)` with 1-step shifted tube schedule |
| `pos_tol`, `ang_tol`, `v_tol` | `1.5`, `0.75`, `0.5` | `run_rmps_episode` | Closed-loop goal-reached tolerances [m], [rad], [m/s] |
| `restore_env_x0` | `True` | `SafeMPDPlanner` | Restores `env.x0` after `planner.plan(x_init, x_goal)` |

### 3.7 Standalone IPOPT NMPC Planner (`NMPCConfig` in [mbd/planners/nmpc_planner.py](../mbd/planners/nmpc_planner.py#L38-L68))
Inherits all `MBDConfig` fields plus:

| Parameter | Default | Meaning |
|---|---|---|
| `mpc_mode` | `False` | `False` = full-horizon optimal control; `True` = receding-horizon closed-loop MPC |
| `mpc_horizon` | `15` | Prediction horizon when `mpc_mode=True` |
| `ipopt_max_iter` / `ipopt_mpc_max_iter` | `400` / `45` | Max IPOPT iterations for open-loop / receding-horizon solves |
| `ipopt_tol`, `ipopt_print_level` | `1e-4`, `0` | IPOPT tolerance and stdout verbosity |
| `w_stage_pos`, `w_stage_heading`, `w_stage_hitch` | `2.0`, `1.5`, `0.5` | Stage tracking & articulation weights |
| `w_terminal_pos`, `w_terminal_heading`, `w_terminal_vel` | `120.0`, `80.0`, `20.0` | Terminal cost weights |
| `w_control`, `w_control_rate`, `w_obs_slack` | `0.05`, `0.4`, `5000.0` | Control effort, smoothness, and soft obstacle slack weights |
| `obstacle_safety_margin`, `hitch_safety_margin_deg` | `0.25`, `1.5` | Clearances [m] and [deg] inside NMPC constraints |

### 3.8 Workspace, Obstacles, Poses & Rendering

| Setting | Where Set | Purpose |
|---|---|---|
| `width`, `height`, `case`, `parking_config`, `resolution` | `Env(...)` ([mbd/envs/env.py](../mbd/envs/env.py#L15-L57)) | World size (`32x32` default), parking-lot geometry dict, obstacle raster resolution |
| `set_rectangle_obs(rects, coordinate_mode, padding)` | `env.set_rectangle_obs(...)` | Sets `[x, y, w, h, angle]` rectangular obstacles (`"center"` or `"left-top"`) |
| `set_circle_obs(circles, padding)` | `env.set_circle_obs(...)` | Sets `[cx, cy, radius]` circular obstacles |
| `set_init_pos(...)`, `set_goal_pos(...)` | Robot instance methods | Initial/goal `(x, y, theta1, theta2[, v, delta])` or parking offsets `(dx, dy)` |
| `render`, `show_animation`, `save_animation`, `save_denoising_animation`, `frame_skip` | `MBDConfig(...)` / `NMPCConfig(...)` | Static rollout figures and Matplotlib/FFmpeg animations |
| `target`, `--mode`, `--no-animate`, `--no-static`, `--save-animation` | [render.py](../render.py#L237-L253) CLI | Post-hoc visualization of `.npy` directories or `.npz` archives |
