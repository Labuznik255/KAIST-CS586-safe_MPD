# Safe-MPD Codebase Documentation

This directory provides a script-by-script technical reference for **Safe Model Predictive Diffusion (Safe-MPD)**, along with concrete extension guides for:
1. **Creating a new vehicle class with more complex dynamics** (e.g., dynamic bicycle/tractor-trailer with tire forces, slip angles, yaw inertia, or higher-dimensional state/control spaces).
2. **Adding perturbations to the existing tractor-trailer models** (process/dynamics disturbances, parameter mismatch, actuator noise, and how they interact with JAX JIT/vmap and shielded rollouts).

---

## Architecture Overview

```
run_diffusion (mbd/planners/mbd_planner.py)
  │
  ├── Config: MBDConfig
  ├── Environment Factory: mbd.envs.get_env(...) (mbd/envs/__init__.py)
  │     ├── Obstacle & Parking Map: Env (mbd/envs/env.py)
  │     └── Robot Dynamics & Safety Class (mbd/robots/)
  │           ├── TractorTrailer2d          (4D state, 2D control)
  │           ├── AccTractorTrailer2d       (6D state, 2D control, inherits TractorTrailer2d)
  │           ├── KinematicBicycle2d        (3D state, 2D control, inherits TractorTrailer2d)
  │           └── NTrailer2d                ((3+N)D state, 2D control, inherits TractorTrailer2d)
  │
  ├── Rollout & Visualization: mbd/utils.py
  │     ├── rollout_us_with_terminal  (jax.lax.scan over env.step + env.get_terminal_reward)
  │     └── create_animation / create_denoising_animation
  │
  └── Evaluation / Benchmarking / Tuning
        ├── mbd/scripts/benchmark_diffusion.py
        ├── tests/test_mbd.py
        ├── tests/stat_mbd.py
        ├── tests/tune_mbd.py
        └── tests/test_planners/ (run_tests.py, test_base.py, test_mbd_planner.py, fixtures/)
```

---

## Index of Script Documentation

### Top-Level
- [render.py](../render.py) → [docs/render.md](render.md) — Standalone plotting script for saved `.npy` rollout outputs.

### Core Package (`mbd/`)
- [mbd/__init__.py](../mbd/__init__.py) → [docs/mbd/__init__.md](mbd/__init__.md)
- [mbd/utils.py](../mbd/utils.py) → [docs/mbd/utils.md](mbd/utils.md) — JAX trajectory rollout helpers (`eval_us`, `rollout_us`, `rollout_us_with_terminal`) and Matplotlib/FFmpeg animation utilities.

#### Environments (`mbd/envs/`)
- [mbd/envs/__init__.py](../mbd/envs/__init__.py) → [docs/mbd/envs/__init__.md](mbd/envs/__init__.md) — `get_env(...)` factory routing `env_name` and `num_trailers` to robot classes.
- [mbd/envs/env.py](../mbd/envs/env.py) → [docs/mbd/envs/env.md](mbd/envs/env.md) — `Env` class defining world bounds, parking-lot layouts, circular/rectangular obstacles.

#### Planners (`mbd/planners/`)
- [mbd/planners/__init__.py](../mbd/planners/__init__.py) → [docs/mbd/planners/__init__.md](mbd/planners/__init__.md)
- [mbd/planners/mbd_planner.py](../mbd/planners/mbd_planner.py) → [docs/mbd/planners/mbd_planner.md](mbd/planners/mbd_planner.md) — `MBDConfig`, JIT cache management, `run_diffusion(...)`, `reverse_once(...)`, steering reward, and visualization.

#### Robot Models (`mbd/robots/`)
- [mbd/robots/__init__.py](../mbd/robots/__init__.py) → [docs/mbd/robots/__init__.md](mbd/robots/__init__.md)
- [mbd/robots/tt2d.py](../mbd/robots/tt2d.py) → [docs/mbd/robots/tt2d.md](mbd/robots/tt2d.md) — `rk4`, `euler`, `State`, and base `TractorTrailer2d` class (kinematic 1-trailer, collision SAT, shielding, projection, guidance, demo generation, rendering).
- [mbd/robots/acc_tt2d.py](../mbd/robots/acc_tt2d.py) → [docs/mbd/robots/acc_tt2d.md](mbd/robots/acc_tt2d.md) — `AccTractorTrailer2d` (6D acceleration-controlled tractor-trailer with predictive rollout-to-stop shielding).
- [mbd/robots/kinematic_bicycle2d.py](../mbd/robots/kinematic_bicycle2d.py) → [docs/mbd/robots/kinematic_bicycle2d.md](mbd/robots/kinematic_bicycle2d.md) — `KinematicBicycle2d` (3D bicycle without trailer).
- [mbd/robots/n_trailer2d.py](../mbd/robots/n_trailer2d.py) → [docs/mbd/robots/n_trailer2d.md](mbd/robots/n_trailer2d.md) — `NTrailer2d` (`(3+N)`D general N-trailer kinematics, collision, and rendering).

#### Scripts (`mbd/scripts/`)
- [mbd/scripts/benchmark_diffusion.py](../mbd/scripts/benchmark_diffusion.py) → [docs/mbd/scripts/benchmark_diffusion.md](mbd/scripts/benchmark_diffusion.md) — Benchmarking harness separating JIT compilation from pure diffusion runtime.

### Tests & Experiments (`tests/`)
- [tests/test_mbd.py](../tests/test_mbd.py) → [docs/tests/test_mbd.md](tests/test_mbd.md) — Minimal custom navigation example.
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

## Extension Guide 1: Creating a New Vehicle Class with More Complex Dynamics

`TractorTrailer2d` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py)) acts as the **de facto base class** for all vehicles (`AccTractorTrailer2d`, `KinematicBicycle2d`, `NTrailer2d` all inherit from it). Follow the pattern of `AccTractorTrailer2d` ([mbd/robots/acc_tt2d.py](../mbd/robots/acc_tt2d.py)) to build a new dynamic vehicle class:

### Step 1: Create the new robot module in `mbd/robots/`
Subclass `TractorTrailer2d` (or `AccTractorTrailer2d`) in e.g. `mbd/robots/dynamic_tt2d.py` and implement/override the following interface required by `mbd_planner.py`, `utils.py`, and `envs/`:

| Component | Required Signature / Property | Why it is needed |
|---|---|---|
| `__init__(self, **kwargs)` | Pop custom physical parameters (mass, inertia, cornering stiffness, etc.) before calling `super().__init__(**kwargs)` | `TractorTrailer2d.__init__` does not accept unknown kwargs; extend `self.x0` and `self.xg` to the new state dimension. |
| `observation_size` (`@property`) | `int` (state dimension $N_x$) | Used in `mbd_planner.py` (`Nx = env.observation_size`). |
| `action_size` (`@property`) | `int` (control dimension $N_u$) | Used in `mbd_planner.py` (`Nu = env.action_size`, `YN` shape `(Hsample, Nu)`). |
| `set_init_pos(...)` / `set_goal_pos(...)` | Override to append/set extra dynamic state entries (e.g., $v_x, v_y, r, \delta$) | Parent sets 4D `[px, py, theta1, theta2]`; subclass appends remaining state dims. |
| `reset(self, rng)` | Returns `State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(self.action_size), jnp.array(False))` | Note `applied_action` shape should match `action_size`. |
| `input_scaler(self, u_normalized)` | Maps normalized `[-1, 1]^{Nu}` to physical units | Called in `_step_internal` and preference penalty. |
| Dynamics ODE + Integrator | e.g. `vehicle_dynamics(self, x, u)` + `rk4(...)` or `euler(...)` | `rk4` is already defined in `mbd/robots/tt2d.py` (currently unused by existing classes, which use `euler`); stiffer dynamic tire/chassis models often benefit from `rk4` or sub-stepping inside `_step_internal`. |
| `step(self, state, action)` & `_step_internal(self, state, action, visualization_mode)` | Returns updated `State` | `mbd_planner.py` JITs `env.step` and directly calls `env._step_internal(state, action, True/False)` when `enable_guidance=True`. |
| `shielded_rollout(self, q, q_proposed)` | Returns `(q_final, obstacle_collision, hitch_violation)` | For dynamic models with momentum, use predictive braking rollout-to-stop (see `AccTractorTrailer2d._check_safety_with_rollout` and `_rollout_step`). |
| Geometry & Safety | `check_obstacle_collision`, `check_hitch_violation` | If the first 4 state dims remain `[px, py, theta1, theta2]`, parent checks work on `q[:4]`. |
| Rewards | `get_reward(self, q)`, `get_terminal_reward(self, q)`, `get_preference_penalty(...)` | Called during step (`get_reward`, `get_preference_penalty`) and at trajectory end by `rollout_us_with_terminal` (`get_terminal_reward`). Add any dynamic regularization (e.g., lateral slip, yaw rate, rollover) here. |
| Demo support (optional) | `generate_demonstration_trajectory`, `eval_xref_logpd` | If `enable_demo=True` is used, slice/pad reference state dimensions as in `AccTractorTrailer2d`. |
| Rendering & Animation | `render`, `setup_animation_patches`, `render_rigid_body`, `update_animation_patches` | Used by `mbd_planner.py` and `utils.create_animation`. |

### Step 2: Export and register the new class
1. **`mbd/robots/__init__.py`**: Import and add the new class to `__all__`.
2. **`mbd/envs/__init__.py`**: Import the class and add an `elif env_name == "<new_name>":` branch in `get_env(...)`, forwarding any new kwargs.
3. **`mbd/planners/mbd_planner.py`**:
   - Add any new configuration parameters (with defaults) to `MBDConfig` (and `dict_to_config_obj` if dict configs are used, plus `__main__` if running standalone).
   - **Check hard-coded action assumptions in `mbd_planner.py`**:
     - `compute_steering_reward(Y0s)` (lines 305–336) assumes `Y0s[:, :, 1]` is steering/steering-rate and multiplies by `env.delta_max`.
     - `reverse_once` (lines 371–392) assumes `Y0s[:, :, 0]` is longitudinal velocity/acceleration for `motion_preference in (2, -2)`.
     - If your new vehicle uses different action ordering or `Nu != 2`, make those functions dispatch on `env` or override them cleanly.
4. **`mbd/utils.py`**:
   - **`rollout_us` and `rollout_us_with_terminal` (lines 30–32, 41–42)** execute a warm-up `state_init = step_env(state, zero_u)` **before** scanning `us`! With a dynamic vehicle (nonzero initial velocity or nonzero drift at `u=0`), this warm-up step advances the state by `dt` under zero input; verify whether that is desired or whether warm-up should only happen if `u` shapes differ.
   - **`create_animation` (lines 398–405)** slices `state_np[:4]` unless `env_name == "kinematic_bicycle2d"` or `num_trailers >= 2`. Ensure `[:4]` represents `[px, py, theta1, theta2]` or update that branch.
5. **Test & Evaluation scripts**:
   - `tests/stat_mbd.py` (`create_trial_environment`, `evaluate_trial_result`), `tests/tune_mbd.py` (`_config_to_dict`, `create_config_with_params`), and `tests/test_planners/fixtures/test_environments.py` (`create_test_tt2d_environment`) pass `MBDConfig` fields explicitly to `mbd.envs.get_env(...)`—forward any new parameters there too.

---

## Extension Guide 2: Adding Perturbations to Existing Tractor-Trailer Models

Depending on whether you want to test **model uncertainty / external disturbances** inside the planner's rollout, or evaluate a **nominal planner against a perturbed plant**, there are four main injection points:

### 1. Continuous-Time Dynamics Perturbations (State/Input/Parameter Disturbances)
- **Where**:
  - `TractorTrailer2d.tractor_trailer_dynamics(self, x, u)` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py#L276-L294))
  - `AccTractorTrailer2d.tractor_trailer_dynamics(self, x, u)` ([mbd/robots/acc_tt2d.py](../mbd/robots/acc_tt2d.py#L90-L116))
  - `NTrailer2d.n_trailer_dynamics(self, x, u)` ([mbd/robots/n_trailer2d.py](../mbd/robots/n_trailer2d.py#L62-L100))
- **Examples**: Additive lateral wind/sideslip disturbance on $\dot{p}_x, \dot{p}_y$, hitch slip/bias on $\dot{\theta}_2$, unmodeled drag/slope on $\dot{v}$, or multiplicative parameter mismatch on `l1`, `l2`, `lh`.
- **Important**: In `AccTractorTrailer2d`, both `_step_internal` and the safety predictor (`shielded_rollout` / `_rollout_step`) call `self.tractor_trailer_dynamics`. If you want the shield to only know the *nominal* model while the *real* transition is perturbed, separate `nominal_dynamics` (used in `shielded_rollout` / `_rollout_step`) from `perturbed_dynamics` (used in the actual step transition).

### 2. Actuator / Input Perturbations
- **Where**:
  - `input_scaler(self, u_normalized)` in `TractorTrailer2d` / `AccTractorTrailer2d` / `NTrailer2d`, OR directly before integration in `_step_internal`.
- **Examples**: Steering bias/offset $\delta + \delta_{\text{bias}}$, velocity scaling $(1 + \epsilon_v) v$, actuator saturation/deadzone.

### 3. Discrete-Time / Stochastic Perturbations (Important JAX PRNG Caveat!)
- **Where**: `_step_internal(self, state, action, visualization_mode)` right after `q_final` (or `q_proposed`) is computed.
- **Critical Architectural Constraint**:
  - Currently, `State` ([mbd/robots/tt2d.py](../mbd/robots/tt2d.py#L33-L40)) has **no `rng` field**:
    ```python
    @struct.dataclass
    class State:
        pipeline_state: jnp.ndarray
        obs: jnp.ndarray
        reward: jnp.ndarray
        done: jnp.ndarray
        applied_action: jnp.ndarray
        in_backup_mode: jnp.ndarray
    ```
  - `env.reset(rng)` discards `rng`, and `step(state, action)` is traced inside `jax.vmap(rollout_us_with_terminal_jit)` and `jax.lax.scan`.
  - Therefore, **do NOT use `np.random` or stateful Python RNG** inside `step` or `dynamics` (JAX traces it once at compile time, freezing the noise).
  - For **stochastic** JAX perturbations inside `step`, add an `rng: jax.Array` field to `State`, initialize it in `reset(self, rng)`, and split it inside `_step_internal` (`rng, step_rng = jax.random.split(state.rng)`), returning `state.replace(..., rng=rng)`. Note: if `Nsample` trajectories are rolled out via `vmap(rollout_us_with_terminal, in_axes=(None, 0))(state_init, Y0s)`, all `Nsample` branches share the same `state_init.rng` unless keys are split per sample in `reverse_once` or perturbations are applied in a separate evaluation rollout.
  - For **deterministic** perturbations (state-dependent, sinusoidal/time-indexed if step index is added to `State`, or constant bias/parameter mismatch), no PRNG change in `State` is needed.

### 4. Closed-Loop / Post-Planning Perturbed Evaluation
- **Where**:
  - Single run: `mbd/planners/mbd_planner.py` post-processing rollout (lines 557–566) where `Y0` is rolled out sequentially via `step_env_jit_final_rollout`.
  - Batch statistical evaluation: `tests/stat_mbd.py` (`create_trial_environment`, `run_statistical_evaluation`, `evaluate_trial_result`).
- **Use case**: Run `run_diffusion` with a nominal `env`, then roll out the planned controls `Y0` through a perturbed dynamics function (or re-plan in MPC fashion) to measure collision/jackknife robustness under model mismatch.
