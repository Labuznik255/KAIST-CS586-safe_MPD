# `mbd/robots/tt2d.py`

**Source file:** [mbd/robots/tt2d.py](../../../mbd/robots/tt2d.py)

## Overview
Implements the **base kinematic single-trailer tractor-trailer model (`TractorTrailer2d`)**, numerical integrators (`rk4`, `euler`), and the `State` Flax dataclass. All other vehicle classes (`AccTractorTrailer2d`, `KinematicBicycle2d`, `NTrailer2d`) inherit from `TractorTrailer2d`.

- **State (4D)**: $x = [p_x, p_y, \theta_1, \theta_2]$ (tractor rear axle $(p_x, p_y)$, tractor heading $\theta_1$, trailer heading $\theta_2$).
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[v, \delta]$ where $v \in [-v_{\max}, v_{\max}]$ and $\delta \in [-\delta_{\max}, \delta_{\max}]$.

---

## Module-Level Functions & Dataclass

### `rk4(dynamics, x, u, dt)` (lines 22–27)
4th-order Runge-Kutta integrator for ODE $\dot{x} = \text{dynamics}(x, u)$ over step `dt`. Currently exported and available for stiffer dynamics (e.g., new dynamic vehicle classes), though existing classes call `euler`.

### `euler(dynamics, x, u, dt)` (lines 29–31)
Forward Euler integrator: $x + \Delta t \cdot \text{dynamics}(x, u)$. Used in `_step_internal` and `AccTractorTrailer2d` rollout steps.

### `@struct.dataclass class State` (lines 33–41)
Immutable JAX-compatible PyTree state container:
- `pipeline_state: jnp.ndarray` — Vehicle state vector $x$.
- `obs: jnp.ndarray` — Observation vector (equal to `pipeline_state`).
- `reward: jnp.ndarray` — Scalar step reward.
- `done: jnp.ndarray` — Termination flag (always `0.0`).
- `applied_action: jnp.ndarray` — Normalized action actually applied after shielding/projection/backup or control perturbation.
- `in_backup_mode: jnp.ndarray` — Boolean flag indicating backup fallback state.
- `rng: jax.Array` — JAX PRNGKey threaded through `reset`, `step`, and `step_perturbed`.

---

## Class: `TractorTrailer2d` (lines 43–1823)

### Initialization & Setup

#### `__init__(self, x0=None, xg=None, env_config=None, case="parking", dt=0.2, H=50, ...)` (lines 49–166)
Stores timing (`dt`, `H`), `num_trailers`, `motion_preference`, safety flags/penalties (`collision_penalty`, `enable_shielded_rollout_collision`, `hitch_penalty`, `enable_shielded_rollout_hitch`, `enable_projection`, `enable_guidance`), `visualization_mode=False`, reward shaping parameters (`reward_threshold`, `ref_reward_threshold`, `max_w_theta`, `hitch_angle_weight`, `theta_max = pi/2`, `phi_max = deg2rad(40.0)`, `d_thr = d_thr_factor * (l1 + lh + l2)`, `k_switch`, `steering_weight`, `preference_penalty_weight`, `heading_reward_weight`, `terminal_reward_threshold`, `terminal_reward_weight`, `ref_*_weight`), vehicle geometry (`l1`, `l2`, `lh`, `lf1`, `lr`, `lf2`, `lr2`, `tractor_width`, `trailer_width`), and input limits (`v_max`, `delta_max`).
Creates or stores `self.env` (`Env`), extracts `self.obs_circles` and `self.obs_rectangles`, initializes `self.x0` and `self.xg`, and initializes `self.xref = None`, `self.rew_xref = None`, and Matplotlib patch attributes.

#### `set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0)` (lines 170–225)
Sets `self.x0` (`4D`):
- In `"parking"` mode with `dx, dy` provided: computes `px = target_x + dx - l1 * cos(theta1)` and `py = entrance_y + dy` relative to the target parking space and parking lot entrance line.
- Otherwise uses explicit `(x, y)` or defaults `(-3.0, 0.0)`.

#### `set_goal_pos(self, x=None, y=None, theta1=None, theta2=None)` (lines 227–253)
Updates only the non-`None` elements of `self.xg` (or initializes from defaults `x=3.0, y=0.0, theta1=pi, theta2=pi`).

#### `set_rectangle_obs(self, rectangles, coordinate_mode="left-top", padding=0.0)` (lines 255–257)
Delegates to `self.env.set_rectangle_obs(...)` and updates `self.obs_rectangles`.

#### `set_circle_obs(self, circles, padding=0.0)` (lines 259–261)
Delegates to `self.env.set_circle_obs(...)` and updates `self.obs_circles`.

#### `reset(self, rng: jax.Array)`
JIT-compiled. Returns `State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(2), jnp.array(False), rng)`.

#### `_init_perturbation_params(...)`, `set_perturbation(...)`, `get_perturbation_info()`
Configures the 4 perturbation levels:
- **Level 0**: Nominal environment (`l1_perturbed, l2_perturbed, lh_perturbed` equal nominal `l1, l2, lh`; zero control/state noise).
- **Level 1**: Gaussian Control Noise ($\mathcal{N}(0, \text{control\_noise\_std}^2)$ added to normalized control commands).
- **Level 2**: Gaussian Model Parameter Uncertainty (time-invariant link lengths `l1_perturbed, l2_perturbed, lh_perturbed` sampled once from $\mathcal{N}(l_{\text{nom}}, (\text{param\_uncertainty\_std} \cdot l_{\text{nom}})^2)$ using `perturbation_seed`, or set explicitly via `perturbed_l1`, `perturbed_l2`, `perturbed_lh`).
- **Level 3**: Non-Gaussian Disturbances (`"uniform"`, `"student_t"`, `"cauchy"`, or `"laplace"` draws applied to controls via `perturb_control` and state derivatives via `perturb_state_dot`).

#### `tractor_trailer_dynamics_perturbed(self, x, u)`
JIT-compiled kinematic tractor-trailer ODE evaluated with time-invariant `self.l1_perturbed, self.l2_perturbed, self.lh_perturbed`.

#### `_sample_non_gaussian(self, rng, shape)`, `perturb_control(self, u_normalized, rng)`, `perturb_state_dot(self, x_dot, rng)`
JIT-compiled perturbation helpers for Levels 1 and 3.

#### `step_perturbed(self, state: State, action: jax.Array) -> State`
JIT-compiled step function for the perturbed simulation rollout. Splits `state.rng`, perturbs normalized control (`perturb_control`), scales control (`input_scaler`), evaluates `tractor_trailer_dynamics_perturbed` and `perturb_state_dot`, integrates over `self.dt`, checks collisions/hitch violations (using perturbed link lengths in Level 2), and returns the updated `State`.

---

### Dynamics, Control Scaling & Stepping

#### `input_scaler(self, u_normalized)` (lines 268–273)
JIT-compiled. Scales normalized `[-1, 1]^2` inputs to `[v, delta] = [u[0] * v_max, u[1] * delta_max]`.
- **Perturbation hook**: Actuator gain/bias or control noise can be injected here (or right before integration in `_step_internal`).

#### `tractor_trailer_dynamics(self, x, u)` (lines 275–294)
JIT-compiled continuous-time kinematic tractor-trailer ODE:
- $\dot{p}_x = v \cos\theta_1$
- $\dot{p}_y = v \sin\theta_1$
- $\dot{\theta}_1 = \frac{v}{l_1}\tan\delta$
- $\dot{\theta}_2 = \frac{v}{l_2}\left(\sin(\theta_1 - \theta_2) - \frac{l_h}{l_1}\cos(\theta_1 - \theta_2)\tan\delta\right)$
- **Perturbation hook**: Primary location for **continuous-time dynamics perturbations** (lateral slip, hitch disturbance, unmodeled drift, or mismatched effective lengths $l_1, l_2, l_h$).

#### `step(self, state: State, action: jax.Array) -> State` (lines 296–299)
JIT-compiled public step API; delegates to `self._step_internal(state, action, self.visualization_mode)`.

#### `_step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State` (lines 301–384)
JIT-compiled with static `visualization_mode` (arg 3). Executes one transition step:
1. Clips `action` to `[-1, 1]`. Checks `backup_active_prev = state.in_backup_mode` and overrides with `u_backup_norm = [0, 0]` if active.
2. Scales `action_eff_norm` via `self.input_scaler`.
3. Dispatches among three safety modes:
   - **`self.enable_projection`**: Calls `project_control_to_safe_set(q, action)`, integrates via `euler(...)`, sets `obstacle_collision = False`, `hitch_violation = False`.
   - **`self.enable_guidance`**: Integrates `q_proposed = euler(...)`, computes `q_reward = self.apply_guidance(q_proposed)`, sets `q_final = q_reward if visualization_mode else q_proposed`, and checks violations on `q_reward`.
   - **Default / Shielded Rollout / Naive Penalty**: Integrates `q_proposed = euler(...)`, then uses `jax.lax.cond` on `enable_shielded_rollout_collision | enable_shielded_rollout_hitch` to run `_step_with_shielded_rollout` or `_step_without_shielded_rollout`. Computes `should_fallback` and `applied_action = where(backup_active_next, u_backup_norm, action)`.
4. Computes `reward = get_reward(q_reward)`, subtracts `collision_penalty` / `hitch_penalty` if violated, and subtracts `get_preference_penalty(u_applied_scaled)` if `motion_preference != 0`.
5. Returns `state.replace(pipeline_state=q_final, obs=q_final, reward=reward, done=0.0, applied_action=applied_action, in_backup_mode=backup_active_prev)` (note: stores `backup_active_prev` rather than `backup_active_next` to avoid locking all future steps into zero control, preserving sample efficiency).
- **Perturbation & Integrator hooks**:
  - Replace `euler(...)` with `rk4(...)` (lines 320, 328, 341) for higher-order integration of complex dynamics.
  - Add discrete-time state perturbations on `q_proposed` / `q_final` here.

#### `_step_with_shielded_rollout(self, data)` (lines 386–390)
JIT-compiled helper unpacking `(q, q_proposed)` and calling `self.shielded_rollout(q, q_proposed)`.

#### `_step_without_shielded_rollout(self, q_proposed)` (lines 392–399)
JIT-compiled helper returning `(q_proposed, obstacle_collision, hitch_violation)` without modifying the state.

#### `shielded_rollout(self, q, q_proposed)` (lines 401–415)
JIT-compiled single-step kinematic shield. Checks `check_obstacle_collision(q_proposed, ...)` and `check_hitch_violation(q_proposed)`. If an enabled violation occurs, reverts `q_final` to the previous safe state `q` (equivalent to braking/stopping in 1 step for kinematic velocity control).

---

### Rewards & Demonstration

#### `get_preference_penalty(self, x_or_u, u=None)` (lines 417–464)
JIT-compiled penalty for moving opposite to `motion_preference` (`1` penalizes $v < 0$, `-1` penalizes $v > 0$, `±2` returns `0.0` since velocity is clipped). Supports both kinematic (`u is None`, `x_or_u` is scaled `[v, delta]`) and acceleration (`len(x_or_u) >= 5`, extracts `v = x_or_u[4]`) conventions.

#### `generate_demonstration_trajectory(self, search_direction="horizontal", num_waypoints_max=4, motion_preference=0)` (lines 466–683)
CPU/NumPy geometric path generator between `x0[:2]` and `xg[:2]`:
- Uses nested `check_line_collision(p1, p2, num_samples=50)` (point-in-circle / point-in-rotated-rectangle checks) to test a direct segment or two perpendicular L-shaped paths.
- Discretizes waypoints into `self.H` points (`xref` of shape `(H, 4)`), sets headings along the final segment (adding $\pi$ for backward preferences `-1, -2`), and creates `self.angle_mask` (`True` only on the final segment). Stores `self.xref` and `self.angle_mask`.

#### `compute_demonstration_reward(self)` (lines 685–703)
`vmap`s `self.get_reward` over `self.xref`, adds `terminal_reward_weight * get_terminal_reward(self.xref[-1])`, and stores `self.rew_xref`.

#### `get_reward(self, q)` (lines 705–793)
JIT-compiled stage reward combining:
- `r_pos`: quadratic clipped distance reward using tractor front axle (if `motion_preference in (1, 2)`) or trailer axle (`get_trailer_position`).
- `r_hdg`: heading alignment for both $\theta_1$ and $\theta_2$ (tests $\pi$-offset wrapping when `motion_preference == 0`).
- `r_art`: articulation reward $1 - (|\theta_1 - \theta_2| / \phi_{\max})^2$.
- Logistic distance blend weight `w_theta = clip(sigmoid((d_thr - d_pos) / k_switch), 0, max_w_theta)`.
- Final reward: `(1 - hitch_angle_weight) * ((1 - w_theta) * r_pos + w_theta * r_hdg) + hitch_angle_weight * r_art`.

#### `get_terminal_reward(self, q)` (lines 795–851)
JIT-compiled terminal position reward $1 - (\text{clip}(d_{\text{pos}}, 0, \text{terminal\_reward\_threshold}) / \text{terminal\_reward\_threshold})^2$.

#### `eval_xref_logpd(self, xs, motion_preference=None)` (lines 1089–1166)
Computes mean log-probability density score comparing rolled-out trajectory `xs` against `self.xref` using nested `@jax.jit def _eval_xref_logpd(...)` weighted by `ref_pos_weight`, `ref_theta1_weight`, `ref_theta2_weight` and masked by `self.angle_mask`.

---

### Geometry, Kinematics & Collision Checking

#### `get_trailer_position(self, x)` (lines 854–867)
JIT-compiled. Returns trailer rear axle center `[trailer_frame_x, trailer_frame_y]` located `lh` behind tractor rear axle along $-\theta_1$ plus `l2` along $-\theta_2$.

#### `get_trailer_back_position(self, x)` (lines 869–882)
JIT-compiled. Returns rear bumper center of the trailer (`l2 + lr2` behind hitch point along $-\theta_2$).

#### `get_tractor_trailer_rectangles(self, x)` (lines 884–936)
JIT-compiled. Computes 4-corner world coordinates, centers, and angles for both tractor and trailer oriented bounding boxes; returns a dict with `tractor_corners`, `tractor_center`, `tractor_angle`, `trailer_corners`, `trailer_center`, `trailer_angle`.

#### `check_rectangle_circle_collision(self, rect_corners, circle_center, circle_radius)` (lines 938–963)
JIT-compiled exact rectangle-vs-circle check by projecting `circle_center` onto each of the 4 rectangle edges (`t in [0, 1]`) and checking `min_dist_sq <= circle_radius ** 2`.

#### `check_rectangle_rectangle_collision(self, rect1_corners, rect2_corners)` (lines 965–1002)
JIT-compiled Separating Axis Theorem (SAT) check across the 4 edge-normal axes of `rect1` and `rect2`.

#### `check_hitch_violation(self, x)` (lines 1004–1010)
JIT-compiled. Wraps $\theta_1 - \theta_2$ to $[-\pi, \pi)$ and returns `abs(hitch_angle) > self.phi_max`.

#### `check_obstacle_collision(self, x, obs_circles, obs_rectangles)` (lines 1012–1080)
JIT-compiled. Checks both tractor and trailer rectangles against all `obs_circles` and `obs_rectangles` via `jax.vmap`.

#### `check_collision(self, x, obs_circles, obs_rectangles)` (lines 1082–1087)
Legacy JIT-compiled wrapper returning `check_obstacle_collision(...) | check_hitch_violation(...)`.

---

### Properties & Visualization

#### `action_size` (`@property`, lines 1168–1170) → `2`
#### `observation_size` (`@property`, lines 1172–1174) → `4`

#### `render(self, ax, xs: jnp.ndarray)` (lines 1176–1276)
Draws parking spaces/numbers, circular/rectangular obstacles, and the planned trajectory `xs[:, :2]` on Matplotlib axis `ax`.

#### `setup_animation_patches(self, ax, tractor_color='blue', trailer_color='red')` (lines 1278–1322)
Creates Matplotlib `Rectangle` patches (`tractor_body`, `trailer_body`, `tractor_wheels`, `trailer_wheels`) and hitch `Line2D` artists (`tractor_hitch_line`, `trailer_hitch_line`).

#### `get_tractor_trailer_positions(self, x)` (lines 1324–1356)
NumPy helper returning axle, front, and hitch coordinates (`tractor_rear`, `tractor_front`, `trailer_rear`, `trailer_front`, `hitch`, `theta1`, `theta2`).

#### `render_rigid_body(self, x, u=None)` (lines 1358–1423)
Computes `Affine2D` Matplotlib transforms for tractor/trailer bodies and wheels (uses `x[5]` for steering if `len(x) >= 6`, else `u[1]`) and hitch line endpoints.

#### `update_animation_patches(self, x, u=None)` (lines 1425–1446)
Applies `render_rigid_body(x, u)` transforms and line data to the Matplotlib patches.

---

### Projection & Gradient Guidance Baselines

#### `_numpy_projection_function(self, args_tuple)` (lines 1448–1685)
CPU NumPy/SciPy (`scipy.optimize.minimize(..., method='trust-constr')`) projection solver invoked via `jax.experimental.io_callback`. Minimizes $\|u - u_{\text{orig}}\|^2$ subject to non-negative signed distance constraints and hitch angle constraints at the predicted next state.
- **Nested helpers**: `tractor_trailer_dynamics_np`, `euler_np` (note: implements RK4), `input_scaler_np`, `get_tractor_trailer_rectangles_np`, `_sdf_oriented_box_np`, `_sdf_two_boxes_np`, `_signed_dist_rect_circle_np`, `_signed_dist_rect_rect_np`, `constraint_function_np`, `objective`, `constraint_function`.

#### `project_control_to_safe_set(self, q_current, u_original_normalized)` (lines 1687–1716)
JIT-compiled wrapper invoking `_numpy_projection_function` via `jax.experimental.io_callback` when `self.enable_projection` is True.

#### `_guidance_function(self, x)` (lines 1718–1781)
JIT-compiled differentiable constraint cost used by `apply_guidance`: penalizes point-to-circle obstacle penetration (`safety_margin = 2.0`) for tractor and trailer centers, plus periodic smooth hitch angle violations (`phi_max`). Ignores rectangular obstacles.

#### `apply_guidance(self, q_proposed, step_size=0.05, max_steps=3)` (lines 1783–1823)
JIT-compiled (`max_steps` static at arg 3). Runs `max_steps` of gradient descent (`jax.grad(self._guidance_function)`) via `jax.lax.scan`, clipping positions to `[-100, 100]` and angles to `[-2*pi, 2*pi]`.
