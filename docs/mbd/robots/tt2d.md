# `mbd/robots/tt2d.py`

**Source file:** [mbd/robots/tt2d.py](../../../mbd/robots/tt2d.py)

## Overview
Implements the **base kinematic single-trailer tractor-trailer model (`TractorTrailer2d`)**, numerical integrators (`rk4`, `euler`), the `State` Flax dataclass (including `rng`), and the 4-level perturbation framework inherited by all other vehicle classes (`AccTractorTrailer2d`, `KinematicBicycle2d`, `NTrailer2d`).

- **State (4D)**: $x = [p_x, p_y, \theta_1, \theta_2]$ (tractor rear axle $(p_x, p_y)$, tractor heading $\theta_1$, trailer heading $\theta_2$).
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[v, \delta]$ where $v \in [-v_{\max}, v_{\max}]$ and $\delta \in [-\delta_{\max}, \delta_{\max}]$.

---

## Module-Level Functions & Dataclass

### `rk4(dynamics, x, u, dt)` (lines 22–27)
4th-order Runge-Kutta integrator for ODE $\dot{x} = \text{dynamics}(x, u)$ over step `dt`.

### `euler(dynamics, x, u, dt)` (lines 29–31)
Forward Euler integrator: $x + \Delta t \cdot \text{dynamics}(x, u)$.

### `@struct.dataclass class State` (lines 34–41)
Immutable JAX-compatible PyTree state container:
- `pipeline_state: jnp.ndarray` — Vehicle state vector $x$.
- `obs: jnp.ndarray` — Observation vector (equal to `pipeline_state`).
- `reward: jnp.ndarray` — Scalar step reward.
- `done: jnp.ndarray` — Termination flag (always `0.0`).
- `applied_action: jnp.ndarray` — Normalized action actually applied after shielding/projection/backup or control perturbation.
- `in_backup_mode: jnp.ndarray` — Boolean flag indicating backup fallback state.
- `rng: jax.Array` — JAX PRNGKey threaded through `reset`, `_step_internal`, and `step_perturbed`.

---

## Class: `TractorTrailer2d` (lines 44–2113)

### Initialization, Pose Setup & Perturbation Configuration

#### `__init__(self, x0=None, xg=None, env_config=None, case="parking", dt=0.2, H=50, ...)` (lines 50–211)
Stores timing (`dt`, `H`), `num_trailers`, `motion_preference`, safety flags/penalties, reward shaping parameters, nominal vehicle geometry (`l1`, `l2`, `lh`, `l1_nominal`, `l2_nominal`, `lh_nominal`, `lf1`, `lr`, `lf2`, `lr2`, `tractor_width`, `trailer_width`), and input limits (`v_max`, `delta_max`). Calls `self._init_perturbation_params(...)` and `self._init_robust_margins(...)` (storing `enable_robust_shielding`, `tube_obstacle_margin`, `tube_hitch_margin`, `enable_ilqr_backup`, `ilqr_iters`), creates/stores `self.env` (`Env`), extracts `obs_circles` and `obs_rectangles`, and initializes `x0` and `xg`.

#### `set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0)` (lines 201–253)
Sets 4D `self.x0` using either parking-lot geometric offsets `(dx, dy)` or direct world coordinates `(x, y)`.

#### `set_goal_pos(self, x=None, y=None, theta1=None, theta2=None)` (lines 258–281)
Updates non-`None` entries of 4D `self.xg`.

#### `_init_perturbation_params(...)` (lines 286–365)
Validates and stores perturbation parameters:
- `perturbation_level` (`0, 1, 2, 3`)
- `perturbation_seed` (generates a fresh seed from `np.random.SeedSequence().entropy` when `None`)
- `control_noise_std` (`jnp.ndarray` of shape `(2,)`)
- `param_uncertainty_std`, `non_gaussian_dist` (`"uniform"`, `"student_t"`, `"cauchy"`, `"laplace"`), `non_gaussian_df`, `state_disturbance_scale`, `disturbance_clip_factor`
- For **Level 2**, samples time-invariant `self.l1_perturbed`, `self.l2_perturbed`, `self.lh_perturbed` from $\mathcal{N}(l_{\text{nom}}, (\sigma_{\text{param}} l_{\text{nom}})^2)$ clipped to $[0.2 l_{\text{nom}}, 5.0 l_{\text{nom}}]$ (or uses explicit `perturbed_l1`, `perturbed_l2`, `perturbed_lh` if provided). For Levels 0, 1, 3, sets them equal to nominal.
- Clears class-level JAX JIT caches on all perturbation/dynamics/collision methods so updated attributes take effect immediately.

#### `set_perturbation(self, level=0, seed=None, ...)` (lines 382–419)
Public helper to configure or change the perturbation mode and noise parameters from any test script. Marks `self._perturbation_manually_set = True`, calls `_init_perturbation_params(...)`, clears `mbd_planner`'s JIT cache, and returns `self.get_perturbation_info()`.

#### Robust Tube Margin Management (lines 421–504)
- **`_parse_margin_array(val)`**, **`_compute_step0_margin(arr, enabled)`**, **`_clear_safety_jit_caches()`**: Normalizes scalar or horizon-array margins (`r_tube`, `b_hitch`), derives the conservative step-0 tracking margin, and clears safety/rollout JIT caches across the class hierarchy.
- **`_init_robust_margins(r_tube=0.0, b_hitch=0.0, enable_robust_shielding=False, enable_ilqr_backup=False, ilqr_iters=4)`**: Stores margin schedules and Stage-2 iLQR flags and invalidates cached JAX graphs.
- **`set_robust_margins(r_tube=0.0, b_hitch=0.0, enable_robust_shielding=True, enable_ilqr_backup=None, ilqr_iters=None)`**: Public helper updating tube-tightened obstacle and hitch margins cleanly at runtime and clearing `mbd_planner`'s JIT cache.

#### `get_perturbation_info(self)` (lines 405–426)
Returns a JSON-serializable dictionary summarizing active perturbation settings, `nominal_params`, and `perturbed_params`.

#### `set_rectangle_obs(self, rectangles, coordinate_mode="left-top", padding=0.0)` (lines 428–430)
Delegates to `self.env.set_rectangle_obs(...)` and updates `self.obs_rectangles`.

#### `set_circle_obs(self, circles, padding=0.0)` (lines 432–434)
Delegates to `self.env.set_circle_obs(...)` and updates `self.obs_circles`.

#### `reset(self, rng: jax.Array)` (lines 437–447)
JIT-compiled. Returns `State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(2), jnp.array(False), rng)`.

---

### Dynamics, Perturbations & Stepping

#### `input_scaler(self, u_normalized)` (lines 450–454)
JIT-compiled. Scales normalized `[-1, 1]^2` inputs to `[v, delta] = [u[0] * v_max, u[1] * delta_max]`.

#### `tractor_trailer_dynamics(self, x, u)` (lines 457–475)
JIT-compiled nominal continuous-time kinematic tractor-trailer ODE using `self.l1`, `self.l2`, `self.lh`.

#### `tractor_trailer_dynamics_perturbed(self, x, u)` (lines 478–493)
JIT-compiled kinematic tractor-trailer ODE using time-invariant `self.l1_perturbed`, `self.l2_perturbed`, `self.lh_perturbed`.

#### `_sample_non_gaussian(self, rng: jax.Array, shape: tuple)` (lines 496–510)
JIT-compiled (`shape` static at arg 2). Samples unit-scale draws from `"uniform"` ($\mathcal{U}(-1, 1)$), `"student_t"` (df `self.non_gaussian_df`), `"cauchy"`, or `"laplace"`, clipped to `±self.disturbance_clip_factor`.

#### `perturb_control(self, u_normalized: jax.Array, rng: jax.Array)` (lines 513–527)
JIT-compiled. Adds $\mathcal{N}(0, \text{control\_noise\_std}^2)$ in Level 1, non-Gaussian noise scaled by `control_noise_std` in Level 3, or zero noise in Levels 0 and 2.

#### `perturb_state_dot(self, x_dot: jax.Array, rng: jax.Array)` (lines 530–540)
JIT-compiled. In Level 3, adds non-Gaussian process disturbance scaled by `state_disturbance_scale` (1.0x on position rates, 0.25x on angular/auxiliary rates) to `x_dot`.

#### `step_perturbed(self, state: State, action: jax.Array) -> State` (lines 543–578)
JIT-compiled perturbed simulation step:
1. Splits `rng, rng_ctrl, rng_state = jax.random.split(state.rng, 3)` so every timestep samples independent random draws.
2. Clips `action` to `[-1, 1]`, applies `perturb_control`, and scales via `input_scaler`.
3. Evaluates `tractor_trailer_dynamics_perturbed` and `perturb_state_dot`, integrating `q_final = q + self.dt * x_dot_pert`.
4. Checks `check_obstacle_collision(..., use_perturbed=(self.perturbation_level == 2))` and `check_hitch_violation(q_final)`, computes reward, and returns updated `State` with `applied_action=u_noisy_norm` and `rng=rng`.

#### `step(self, state: State, action: jax.Array) -> State` (lines 581–583)
JIT-compiled nominal planning step; delegates to `self._step_internal(state, action, self.visualization_mode)`.

#### `_step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State` (lines 586–669)
JIT-compiled (`visualization_mode` static at arg 3). Nominal step supporting projection, guidance, shielded rollout, and naive penalty modes, splitting `state.rng` for the returned `State`.

#### `_step_with_shielded_rollout(self, data)` (lines 672–675) & `_step_without_shielded_rollout(self, q_proposed)` (lines 678–684)
JIT-compiled helpers for conditional shielded vs. unshielded nominal rollout.

#### `shielded_rollout(self, q, q_proposed)` (lines 687–700)
JIT-compiled single-step kinematic shield reverting `q_final` to `q` on enabled collision or hitch violation.

---

### Rewards, Demonstrations, Geometry & Collision Checking

- **`get_preference_penalty(self, x_or_u, u=None)`** (lines 703–749): Penalizes motion opposite to `motion_preference`.
- **`generate_demonstration_trajectory(...)`** (lines 751–968) & **`compute_demonstration_reward(self)`** (lines 970–988): Waypoint-based reference trajectory generator and reward evaluator.
- **`get_reward(self, q)`** (lines 991–1078) & **`get_terminal_reward(self, q)`** (lines 1081–1136): Stage and terminal rewards.
- **`get_trailer_position(self, x)`** (lines 1140–1152) & **`get_trailer_back_position(self, x)`** (lines 1155–1167): Trailer axle and rear bumper positions.
- **`get_tractor_trailer_rectangles(self, x, use_perturbed: bool = False)`** (lines 1170–1224): Computes oriented bounding box corners using nominal `(l1, l2, lh)` or perturbed `(l1_perturbed, l2_perturbed, lh_perturbed)` when `use_perturbed=True`.
- **`check_rectangle_circle_collision`** (line 1336), **`check_rectangle_rectangle_collision`** (line 1363), **`check_hitch_violation(self, x, hitch_margin: float = 0.0)`** (lines 1402–1409, checking `|hitch_angle| > max(0.0, self.phi_max - hitch_margin)`), **`check_obstacle_collision(self, x, obs_circles, obs_rectangles, use_perturbed: bool = False, obstacle_margin: float = 0.0)`** (lines 1411–1488, inflating active circle radii by `obstacle_margin` and active rectangle width/height by `2 * obstacle_margin`), **`check_collision`** (line 1490).
- **`eval_xref_logpd(self, xs, motion_preference=None)`** (lines 1377–1454).

---

### Properties, Rendering, Projection & Guidance

- **`action_size`** (`@property`, lines 1457–1458) → `2`; **`observation_size`** (`@property`, lines 1461–1462) → `4`.
- **`render(self, ax, xs)`** (lines 1464–1562), **`setup_animation_patches(self, ax, ...)`** (lines 1566–1612, resets `self.tractor_wheels = []` and `self.trailer_wheels = []`), **`get_tractor_trailer_positions(self, x)`** (lines 1614–1646), **`render_rigid_body(self, x, u=None)`** (lines 1648–1713), **`update_animation_patches(self, x, u=None)`** (lines 1715–1736).
- **`_numpy_projection_function`** (lines 1738–1975), **`project_control_to_safe_set`** (lines 1978–2006), **`_guidance_function`** (lines 2009–2071), **`apply_guidance`** (lines 2074–2113).
