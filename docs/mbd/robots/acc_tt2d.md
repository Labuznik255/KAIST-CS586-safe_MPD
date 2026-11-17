# `mbd/robots/acc_tt2d.py`

**Source file:** [mbd/robots/acc_tt2d.py](../../../mbd/robots/acc_tt2d.py)

## Overview
Implements `AccTractorTrailer2d`, an acceleration- and steering-rate-controlled subclass of `TractorTrailer2d`:
- **State (6D)**: $x = [p_x, p_y, \theta_1, \theta_2, v, \delta]$
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[a, \omega] = [\dot{v}, \dot{\delta}]$ where $a \in [-a_{\max}, a_{\max}]$ and $\omega \in [-\omega_{\max}, \omega_{\max}]$.
- **Key Addition**: Predictive **rollout-to-stop** safety verification inside `shielded_rollout` (`_check_safety_with_rollout` + `_rollout_step`), because nonzero velocity $v$ means a single step cannot stop instantaneously.

> **Template Note**: `AccTractorTrailer2d` is the cleanest reference template in this repository for creating a **new vehicle class with more complex dynamics** (6D+ state, actuator dynamics, and multi-step predictive shielding).

---

## Class: `AccTractorTrailer2d(TractorTrailer2d)` (lines 19–433)

### `__init__(self, **kwargs)` (lines 26–53)
1. Pops `a_max` (default `2.0` m/s²) and `omega_max` (default `1.0` rad/s) from `kwargs`, then calls `super().__init__(**kwargs)`.
2. Pre-computes JIT-constant rollout parameters: `_rollout_dt = 1.0`, `_v_threshold = 0.01`, `_max_rollout_steps = int(2.0 * v_max / (a_max * _rollout_dt))`.
3. Extends `self.x0` and `self.xg` from 4D to 6D by appending `[0.0, 0.0]` (`v=0, delta=0`).

### `set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0, v=0.0, delta=0.0)` (lines 55–64)
Calls `super().set_init_pos(...)` for 4D pose and concatenates `[v, delta]` to form 6D `self.x0`.

### `set_goal_pos(self, x=None, y=None, theta1=None, theta2=None, v=0.0, delta=0.0)` (lines 66–75)
Calls `super().set_goal_pos(...)` for 4D pose and concatenates `[v, delta]` to form 6D `self.xg`.

### `reset(self, rng: jax.Array)`
JIT-compiled. Returns `State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(2), jnp.array(False), rng)`.

### `tractor_trailer_dynamics_perturbed(self, x, u)` & `step_perturbed(self, state: State, action: jax.Array) -> State`
6D acceleration-controlled perturbed dynamics (using `self.l1_perturbed, self.l2_perturbed, self.lh_perturbed`) and perturbed simulation step supporting Perturbation Levels 0, 1, 2, and 3.

### `input_scaler(self, u_normalized)` (lines 82–87)
JIT-compiled. Maps `u_normalized` to `[u[0] * a_max, u[1] * omega_max]`.

### `tractor_trailer_dynamics(self, x, u)` (lines 89–116)
JIT-compiled 6D continuous-time ODE:
- Clips `v` to `[-v_max, v_max]` and `delta` to `[-delta_max, delta_max]`.
- $\dot{p}_x, \dot{p}_y, \dot{\theta}_1, \dot{\theta}_2$: same kinematic equations as `TractorTrailer2d`, evaluated at clipped `(v, delta)`.
- $\dot{v} = a$, $\dot{\delta} = \omega$.
- **Perturbation hook**: Add acceleration/tire-drag/sideslip/hitch-disturbance terms here (or separate nominal vs. perturbed dynamics if the safety shield should not know the perturbation).

### `step(self, state: State, action: jax.Array) -> State` (lines 118–121)
JIT-compiled wrapper calling `_step_internal(state, action, self.visualization_mode)`.

### `_step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State` (lines 123–210)
JIT-compiled (`visualization_mode` static at arg 3):
- Computes backup deceleration control `u_backup_norm` aiming to brake `v_curr` toward zero (`a_backup = clip(|v_curr / dt|, 0, a_max)`, `omega = 0`).
- Integrates via `euler(self.tractor_trailer_dynamics, q, u_scaled, self.dt)`.
- Supports `enable_projection`, `enable_guidance`, shielded rollout (`_step_with_shielded_rollout`), and naive penalty (`_step_without_shielded_rollout`).
- Evaluates `get_reward(q_reward[:4])`, applies collision/hitch penalties, and applies `get_preference_penalty(q_final, u_applied_scaled)` (using `q_final[4]` for velocity).

### `_step_with_shielded_rollout(self, data)` (lines 212–216)
Calls `self.shielded_rollout(q, q_proposed)`.

### `_step_without_shielded_rollout(self, q_proposed)` (lines 218–225)
Checks obstacle collision and hitch violation on `q_proposed[:4]` without altering `q_proposed`.

### `shielded_rollout(self, q, q_proposed)` (lines 227–266)
JIT-compiled predictive shield:
1. Checks `abs(q_proposed[4]) <= _v_threshold`:
   - If stopped/slow: calls `_check_safety_single_step`.
   - If moving: calls `_check_safety_with_rollout` to simulate braking to rest.
2. Computes 1-step backup control `u_backup_scaled = [decel_sign * a_backup, 0.0]` and `q_backup = euler(self.tractor_trailer_dynamics, q, u_backup_scaled, self.dt)`.
3. Returns `q_backup` if an enabled violation is detected, else `q_proposed`, along with `(obstacle_collision, hitch_violation)`.

### `_check_safety_single_step(self, data)` (lines 268–274)
Checks `check_obstacle_collision(q_new[:4], ...)` and `check_hitch_violation(q_new[:4])`.

### `_rollout_step(self, carry, _)` (lines 276–297)
Single step of the `jax.lax.scan` braking rollout: if `|v_curr| > _v_threshold`, integrates `q_curr` with `u_stop` over `_rollout_dt`, checks collision/hitch on `q_next[:4]`, and OR-accumulates violations.

### `_check_safety_with_rollout(self, data)` (lines 299–319)
Checks `q_new` first, computes braking control `u_stop`, and scans `_rollout_step` for `_max_rollout_steps` iterations, returning `(obstacle_collision_any, hitch_violation_any)`.

### `project_to_safe_set(self, q_prop)` (lines 321–341)
Projects `q_prop[:4]` via `super().project_to_safe_set` (note: parent implements control projection, not state projection) and reattaches `q_prop[4:]`.

### `project_control_to_safe_set(self, q_current, u_original_normalized)` (lines 343–358)
Delegates `q_current[:4]` to `super().project_control_to_safe_set`.

### `apply_guidance(self, q_proposed, step_size=0.05, max_steps=3)` (lines 360–378)
Applies parent 4D guidance to `q_proposed[:4]` and reattaches `q_proposed[4:]`.

### `get_preference_penalty(self, x, u)` (lines 380–385)
Delegates to `super().get_preference_penalty(x, u)`.

### `get_reward(self, q)` (lines 387–389)
Delegates to `super().get_reward(q[:4])`.

### `get_terminal_reward(self, q)` (lines 391–393)
Delegates to `super().get_terminal_reward(q[:4])`.

### `action_size` (`@property`, lines 395–397) → `2`
### `observation_size` (`@property`, lines 399–401) → `6`

### `generate_demonstration_trajectory(...)` (lines 403–415)
Calls `super().generate_demonstration_trajectory(...)` to produce 4D `self.xref`.

### `compute_demonstration_reward(self)` (lines 417–422)
Calls `super().compute_demonstration_reward()`.

### `eval_xref_logpd(self, xs, motion_preference=None)` (lines 424–433)
Slices `xs[:, :4]` and delegates to `super().eval_xref_logpd(...)`.

---

## Extension Notes
- **Building a more complex dynamic vehicle**: Subclass `AccTractorTrailer2d` (or `TractorTrailer2d` following this file's structure). You will keep the 4D geometric prefix `[px, py, theta1, theta2]` for obstacle/hitch/reward reuse, append dynamic states (e.g., `[vx, vy, r1, r2, delta]`), update `observation_size` / `action_size`, replace `tractor_trailer_dynamics` (and optionally use `rk4` instead of `euler`), and update the braking backup policy inside `shielded_rollout` and `_check_safety_with_rollout`.
- **Adding perturbations**: Inject ODE perturbations in `tractor_trailer_dynamics` (lines 90–116) or post-integration perturbations in `_step_internal` (line 167). Keep in mind that `_rollout_step` (line 284) and `shielded_rollout` (line 260) also call `self.tractor_trailer_dynamics`—decide whether the safety shield uses the perturbed or nominal ODE.
