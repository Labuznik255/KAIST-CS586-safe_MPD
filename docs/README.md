# Safe-MPD Codebase Documentation

This directory provides a script-by-script technical reference for **Safe Model Predictive Diffusion (Safe-MPD)**, covering:
1. Every Python script, class, and function in the repository.
2. The implemented **4-Level Perturbation Simulation Framework** across all vehicle models (`TractorTrailer2d`, `AccTractorTrailer2d`, `KinematicBicycle2d`, `NTrailer2d`), standalone perturbation rollout testing (`tests/test_perturbations.py`), and standalone visualization (`render.py`).
3. **Robust Safe-MPD (RMPS, Stages 1–5)** for `AccTractorTrailer2d`: offline reachable tube calibration (`mbd/robust/tube_calibration.py`), tube constraint tightening + optional Stage-2 vectorized JAX iLQR backup (`mbd/robust/ilqr_backup.py`), NumPy planner wrapper (`mbd/robust/planner_wrapper.py`), online CPU CasADi Tube-Based NMPC shield (`mbd/robust/nmpc_backup.py`), closed-loop RMPS execution & replanning (`mbd/robust/rmps_runner.py`), and comparative benchmarking (`tests/test_robust_rmps.py`).
4. A complete **Testing, Benchmarking, and Parameter Reference Guide** ([docs/TESTING_AND_BENCHMARKING.md](TESTING_AND_BENCHMARKING.md)) plus extension guides for **creating a new vehicle class** and **customizing robust shielding**.

---

## Architecture Overview

```
run_diffusion (mbd/planners/mbd_planner.py)
  │
  ├── Config: MBDConfig (includes perturbation_level 0..3 & noise/uncertainty params)
  ├── Environment Factory: mbd.envs.get_env(...) (mbd/envs/__init__.py)
  │     ├── Obstacle & Parking Map: Env (mbd/envs/env.py)
  │     └── Robot Dynamics, Perturbations & Safety Class (mbd/robots/)
  │           ├── TractorTrailer2d          (4D state, 2D control; step & step_perturbed)
  │           ├── AccTractorTrailer2d       (6D state, 2D control, inherits TractorTrailer2d)
  │           ├── KinematicBicycle2d        (3D state, 2D control, inherits TractorTrailer2d)
  │           └── NTrailer2d                ((3+N)D state, 2D control, inherits TractorTrailer2d)
  │
  ├── Rollout, Metadata & Visualization: mbd/utils.py
  │     ├── rollout_us_with_terminal  (nominal planning rollout + terminal reward)
  │     ├── rollout_perturbed_us      (perturbed execution rollout via env.step_perturbed)
  │     ├── save_env_metadata / load_env_from_results (env_metadata.json persistence)
  │     └── create_animation / create_denoising_animation
  │
  ├── Standalone Post-Planning & RMPS Scripts
  │     ├── tests/test_perturbations.py (perturb already-generated final_actions.npy)
  │     ├── render.py                   (standalone static + animated visualizer)
  │     └── mbd/robust/                 (RMPS Stages 1–4: offline tubes, iLQR, SafeMPDPlanner, NMPC shield, online runner)
  │
  └── Evaluation / Benchmarking / Tuning
        ├── mbd/scripts/benchmark_diffusion.py
        ├── tests/test_mbd.py
        ├── tests/test_nmpc.py
        ├── tests/test_robust_rmps.py   (RMPS Stage 5: 4-variant Level 0–3 benchmark)
        ├── tests/stat_mbd.py
        ├── tests/tune_mbd.py
        └── tests/test_planners/ (run_tests.py, test_base.py, test_mbd_planner.py, fixtures/)
```

---

## Guides
- **[Testing, Benchmarking, and Parameter Reference Guide](TESTING_AND_BENCHMARKING.md)** — How to run all tests, tube calibration, and benchmarks across settings, plus a complete table of every configurable parameter and where to set it.

---

## Index of Script Documentation

### Top-Level
- [render.py](../render.py) → [docs/render.md](render.md) — Standalone static + animated visualizer for any generated trajectory directory or `.npz` archive.

### Core Package (`mbd/`)
- [mbd/__init__.py](../mbd/__init__.py) → [docs/mbd/__init__.md](mbd/__init__.md)
- [mbd/utils.py](../mbd/utils.py) → [docs/mbd/utils.md](mbd/utils.md) — JAX trajectory rollout helpers (`eval_us`, `rollout_us`, `rollout_us_with_terminal`, `rollout_perturbed_us`), environment metadata persistence (`save_env_metadata`, `load_env_from_results`), and Matplotlib/FFmpeg animation utilities.

#### Environments (`mbd/envs/`)
- [mbd/envs/__init__.py](../mbd/envs/__init__.py) → [docs/mbd/envs/__init__.md](mbd/envs/__init__.md) — `get_env(...)` factory routing `env_name`, `num_trailers`, and perturbation parameters to robot classes.
- [mbd/envs/env.py](../mbd/envs/env.py) → [docs/mbd/envs/env.md](mbd/envs/env.md) — `Env` class defining world bounds, parking-lot layouts, circular/rectangular obstacles.

#### Planners (`mbd/planners/`)
- [mbd/planners/__init__.py](../mbd/planners/__init__.py) → [docs/mbd/planners/__init__.md](mbd/planners/__init__.md)
- [mbd/planners/mbd_planner.py](../mbd/planners/mbd_planner.py) → [docs/mbd/planners/mbd_planner.md](mbd/planners/mbd_planner.md) — `MBDConfig`, JIT cache management, `run_diffusion(...)`, `reverse_once(...)`, nominal and perturbed trajectory rollouts, and result saving/rendering.

#### Robot Models (`mbd/robots/`)
- [mbd/robots/__init__.py](../mbd/robots/__init__.py) → [docs/mbd/robots/__init__.md](mbd/robots/__init__.md)
- [mbd/robots/tt2d.py](../mbd/robots/tt2d.py) → [docs/mbd/robots/tt2d.md](mbd/robots/tt2d.md) — `rk4`, `euler`, `State` (with `rng`), and base `TractorTrailer2d` class (kinematic 1-trailer, 4-level perturbation framework, collision SAT, shielding, projection, guidance, demo generation, rendering).
- [mbd/robots/acc_tt2d.py](../mbd/robots/acc_tt2d.py) → [docs/mbd/robots/acc_tt2d.md](mbd/robots/acc_tt2d.md) — `AccTractorTrailer2d` (6D acceleration-controlled tractor-trailer with predictive rollout-to-stop shielding and 6D perturbed rollout).
- [mbd/robots/kinematic_bicycle2d.py](../mbd/robots/kinematic_bicycle2d.py) → [docs/mbd/robots/kinematic_bicycle2d.md](mbd/robots/kinematic_bicycle2d.md) — `KinematicBicycle2d` (3D bicycle without trailer, nominal and perturbed dynamics/collision).
- [mbd/robots/n_trailer2d.py](../mbd/robots/n_trailer2d.py) → [docs/mbd/robots/n_trailer2d.md](mbd/robots/n_trailer2d.md) — `NTrailer2d` (`(3+N)`D general N-trailer kinematics, per-trailer length uncertainty, collision, and rendering).

#### Robust Safe-MPD / RMPS (`mbd/robust/`)
- [mbd/robust/__init__.py](../mbd/robust/__init__.py) → [docs/mbd/robust/__init__.md](mbd/robust/__init__.md) — Public exports for Stages 1–4.
- [mbd/robust/tube_calibration.py](../mbd/robust/tube_calibration.py) → [docs/mbd/robust/tube_calibration.md](mbd/robust/tube_calibration.md) — **Stage 1**: Offline TV-LQR + Monte Carlo reachable tube calibration (`r_tube`, `b_hitch`, `b_v`).
- [mbd/robust/ilqr_backup.py](../mbd/robust/ilqr_backup.py) → [docs/mbd/robust/ilqr_backup.md](mbd/robust/ilqr_backup.md) — **Stage 2**: Pure-JAX branchless vectorized constrained iLQR backup solver.
- [mbd/robust/planner_wrapper.py](../mbd/robust/planner_wrapper.py) → [docs/mbd/robust/planner_wrapper.md](mbd/robust/planner_wrapper.md) — **Stage 3**: NumPy-facing `SafeMPDPlanner` wrapper around `run_diffusion`.
- [mbd/robust/nmpc_backup.py](../mbd/robust/nmpc_backup.py) → [docs/mbd/robust/nmpc_backup.md](mbd/robust/nmpc_backup.md) — **Stage 3**: CPU CasADi + IPOPT Tube-Based Recovery & Tracking NMPC shield (`RobustNMPCBackup`).
- [mbd/robust/rmps_runner.py](../mbd/robust/rmps_runner.py) → [docs/mbd/robust/rmps_runner.md](mbd/robust/rmps_runner.md) — **Stage 4**: Online closed-loop RMPS execution, replanning (`run_rmps_episode`), and diagnostic plots (`plot_rmps_diagnostics`).

#### Scripts (`mbd/scripts/`)
- [mbd/scripts/benchmark_diffusion.py](../mbd/scripts/benchmark_diffusion.py) → [docs/mbd/scripts/benchmark_diffusion.md](mbd/scripts/benchmark_diffusion.md) — Benchmarking harness separating JIT compilation from pure diffusion runtime.

### Tests & Experiments (`tests/`)
- [tests/test_mbd.py](../tests/test_mbd.py) → [docs/tests/test_mbd.md](tests/test_mbd.md) — Minimal custom navigation example with perturbation configuration.
- [tests/test_perturbations.py](../tests/test_perturbations.py) → [docs/tests/test_perturbations.md](tests/test_perturbations.md) — Standalone script that takes an already-generated nominal action sequence (`final_actions.npy`) and perturbs it under Levels 0–3 without re-running diffusion.
- [tests/test_robust_rmps.py](../tests/test_robust_rmps.py) → [docs/tests/test_robust_rmps.md](tests/test_robust_rmps.md) — **Stage 5**: Comparative 4-variant RMPS benchmark (`baseline`, `tube_only`, `rmps_no_replan`, `rmps_replan`) across Levels 0–3.
- [tests/test_nmpc.py](../tests/test_nmpc.py) — Standalone CasADi + IPOPT NMPC planner verification across all 4 vehicles and perturbation levels.
- [tests/stat_mbd.py](../tests/stat_mbd.py) → [docs/tests/stat_mbd.md](tests/stat_mbd.md) — Batch Monte Carlo evaluator, metrics, and spatial heat map plotting.
- [tests/tune_mbd.py](../tests/tune_mbd.py) → [docs/tests/tune_mbd.md](tests/tune_mbd.md) — Optuna GP + MedianPruner hyperparameter optimizer with W&B logging.
- [tests/test_planners/__init__.py](../tests/test_planners/__init__.py) → [docs/tests/test_planners/__init__.md](tests/test_planners/__init__.md)
- [tests/test_planners/run_tests.py](../tests/test_planners/run_tests.py) → [docs/tests/test_planners/run_tests.md](tests/test_planners/run_tests.md) — CLI test runner (`--all`, `--demo`, `--acc`, `--kinematic`, `--single`, `--compare`, `--list`).
- [tests/test_planners/test_base.py](../tests/test_planners/test_base.py) → [docs/tests/test_planners/test_base.md](tests/test_planners/test_base.md) — `BaseMBDTest` (`unittest.TestCase` subclass) and `TimeoutContext`.
- [tests/test_planners/test_mbd_planner.py](../tests/test_planners/test_mbd_planner.py) → [docs/tests/test_planners/test_mbd_planner.md](tests/test_planners/test_mbd_planner.md) — `TestMBDPlanner` scenario and utility unit tests.
- [tests/test_planners/fixtures/__init__.py](../tests/test_planners/fixtures/__init__.py) → [docs/tests/test_planners/fixtures/__init__.md](tests/test_planners/fixtures/__init__.md)
- [tests/test_planners/fixtures/test_configs.py](../tests/test_planners/fixtures/test_configs.py) → [docs/tests/test_planners/fixtures/test_configs.md](tests/test_planners/fixtures/test_configs.md) — `TestConfig` dataclass and predefined scenario dictionaries.
- [tests/test_planners/fixtures/test_environments.py](../tests/test_planners/fixtures/test_environments.py) → [docs/tests/test_planners/fixtures/test_environments.md](tests/test_planners/fixtures/test_environments.md) — Environment construction helpers for tests.

---

## Perturbation Simulation Framework (Levels 0–3)

All four robot classes support four perturbation modes controlled via `MBDConfig`, `mbd.envs.get_env(...)`, `env.set_perturbation(...)`, or [tests/test_perturbations.py](../tests/test_perturbations.py):

| Level | Name | Implementation Details |
|---|---|---|
| **Level 0** | **Nominal Environment** | Perfect model knowledge (`l1_perturbed = l1`, `l2_perturbed = l2`, `lh_perturbed = lh`), zero control noise, zero state-rate disturbances. |
| **Level 1** | **Gaussian Control Noise** | At every step $t$, `state.rng` is split and independent Gaussian noise $\epsilon_u \sim \mathcal{N}(0, \text{diag}(\sigma_{u,0}^2, \sigma_{u,1}^2))$ (`control_noise_std`) is added to normalized control commands in `perturb_control` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py#L513-L527)). |
| **Level 2** | **Gaussian Model Parameter Uncertainty** | Time-invariant link lengths (`l1_perturbed`, `l2_perturbed`, `lh_perturbed`, and `trailer_lengths_perturbed` for `NTrailer2d`) are sampled once before the rollout from $\mathcal{N}(l_{\text{nom}}, (\text{param\_uncertainty\_std} \cdot l_{\text{nom}})^2)$ (or overridden via `perturbed_l1`, `perturbed_l2`, `perturbed_lh`) in `_init_perturbation_params` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py#L286-L365)) and used in `tractor_trailer_dynamics_perturbed` and collision geometry (`use_perturbed=True`). |
| **Level 3** | **Non-Gaussian Disturbances** | At every step $t$, non-Gaussian draws from `"student_t"` (heavy-tailed with `non_gaussian_df`), `"cauchy"`, `"laplace"`, or `"uniform"` are sampled in `_sample_non_gaussian` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py#L496-L510)) and applied to both control inputs (`perturb_control`) and state derivatives (`perturb_state_dot` scaled by `state_disturbance_scale`). |

### Saved Results Structure (`results/{env_name}/`)
Every planning run (`run_diffusion`) and standalone perturbation run (`tests/test_perturbations.py`) saves:
1. `nominal_trajectory_states.npy` (and `trajectory_states.npy`): Nominal trajectory states $(H+1, N_x)$.
2. `perturbed_trajectory_states.npy`: Perturbed trajectory states $(H+1, N_x)$.
3. `final_actions.npy`: Planned nominal controls $(H, N_u)$.
4. `perturbed_actions.npy`: Perturbed controls actually applied during the perturbed rollout $(H, N_u)$.
5. `mu_0ts.npy`: Intermediate diffusion denoising means $(N_{\text{diffuse}}-1, H, N_u)$.
6. `env_metadata.json`: Full environment/obstacle/perturbation metadata so `tests/test_perturbations.py` and `render.py` can reconstruct the exact environment without re-running diffusion.

---

## Extension Guide 1: Creating a New Vehicle Class with More Complex Dynamics

`TractorTrailer2d` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py)) is the base class for all vehicles (`AccTractorTrailer2d`, `KinematicBicycle2d`, `NTrailer2d`). Follow `AccTractorTrailer2d` ([mbd/robots/acc_tt2d.py](../mbd/robots/acc_tt2d.py)) to build a new dynamic vehicle class:

| Component | Required Signature / Property | Notes |
|---|---|---|
| `__init__(self, **kwargs)` | Pop custom physical parameters before `super().__init__(**kwargs)` | Extend `self.x0` and `self.xg` to the new state dimension $N_x$. |
| `observation_size` / `action_size` | `@property` returning `int` ($N_x$, $N_u$) | Used by `mbd_planner.py` (`Nx`, `Nu`). |
| `set_init_pos` / `set_goal_pos` | Override to append extra dynamic states | Keep `[px, py, theta1, theta2]` as the first 4 state components to reuse collision, hitch, and reward methods. |
| `reset(self, rng)` | Returns `State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(self.action_size), jnp.array(False), rng)` | Threads `rng` into `State`. |
| `input_scaler(self, u_normalized)` | Maps `[-1, 1]^{Nu}` to physical units | Used in both `_step_internal` and `step_perturbed`. |
| Nominal & Perturbed ODEs | `tractor_trailer_dynamics(self, x, u)` and `tractor_trailer_dynamics_perturbed(self, x, u)` | Use nominal parameters in `tractor_trailer_dynamics` and `self.l1_perturbed` etc. in `tractor_trailer_dynamics_perturbed`. Use `rk4` or `euler`. |
| `step`, `_step_internal`, `step_perturbed` | `(state: State, action: jax.Array) -> State` | `_step_internal` is used during diffusion planning; `step_perturbed` is used for the perturbed rollout. |
| `shielded_rollout(self, q, q_proposed)` | Returns `(q_final, obstacle_collision, hitch_violation)` | For dynamic vehicles with momentum, use predictive rollout-to-stop as in `AccTractorTrailer2d._check_safety_with_rollout`. |

Then register the new class in:
1. [mbd/robots/__init__.py](../mbd/robots/__init__.py)
2. [mbd/envs/__init__.py](../mbd/envs/__init__.py) (`get_env`)
3. [mbd/utils.py](../mbd/utils.py) (`save_env_metadata` / `load_env_from_results` mapping)

---

## Extension Guide 2: Implementing Robust Shielding for the Rollout

Robust shielding (Stages 1–5) is already implemented for `AccTractorTrailer2d` under [mbd/robust/](../mbd/robust/) and can be extended to additional robot classes:
- **Planning-Layer Tube Tightening & GPU Fallback**:
  - `TractorTrailer2d` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py)) provides `check_obstacle_collision(..., obstacle_margin=...)`, `check_hitch_violation(..., hitch_margin=...)`, and `set_robust_margins(...)`.
  - `AccTractorTrailer2d` ([mbd/robots/acc_tt2d.py](../mbd/robots/acc_tt2d.py)) applies time-indexed tube schedules across `_rollout_step` / `_check_safety_with_rollout` (Stage 1 braking) and falls back to `solve_ilqr_backup` ([mbd/robust/ilqr_backup.py](../mbd/robust/ilqr_backup.py)) when `enable_ilqr_backup=True` (Stage 2).
- **Online Runtime Shielding & Replanning**:
  - `RobustNMPCBackup` ([mbd/robust/nmpc_backup.py](../mbd/robust/nmpc_backup.py)) and `run_rmps_episode` ([mbd/robust/rmps_runner.py](../mbd/robust/rmps_runner.py)) monitor recoverability online against `step_perturbed`, execute tube-based tracking NMPC stops when threatened, and trigger `SafeMPDPlanner.plan(...)` replanning from stopped states.
