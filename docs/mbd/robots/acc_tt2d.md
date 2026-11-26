# `mbd/robots/acc_tt2d.py`

**Source file:** [mbd/robots/acc_tt2d.py](../../../mbd/robots/acc_tt2d.py)

## Overview
Implements `AccTractorTrailer2d`, an acceleration- and steering-rate-controlled subclass of `TractorTrailer2d`:
- **State (6D)**: $x = [p_x, p_y, \theta_1, \theta_2, v, \delta]$
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[a, \omega] = [\dot{v}, \dot{\delta}]$ where $a \in [-a_{\max}, a_{\max}]$ and $\omega \in [-\omega_{\max}, \omega_{\max}]$.
- **Key Features**: Predictive **rollout-to-stop** safety verification inside `shielded_rollout` with optional **RMPS tube constraint tightening** (`_r_margin_rollout`, `_b_margin_rollout`) and **Stage-2 branchless JAX iLQR backup fallback** (`solve_ilqr_backup`), plus 6D perturbed dynamics (`tractor_trailer_dynamics_perturbed`) and `step_perturbed` for Levels 0–3 perturbations.

---

## Class: `AccTractorTrailer2d(TractorTrailer2d)` (lines 20–634)

### `__init__(self, **kwargs)` (lines 27–58)
Pops `a_max` (default `2.0` m/s²) and `omega_max` (default `1.0` rad/s), calls `super().__init__(**kwargs)` (initializing perturbation and robust margin attributes), sets `_rollout_dt = 1.0`, `_v_threshold = 0.01`, `_max_rollout_steps`, `_ilqr_horizon = 10`, `_ilqr_dt = 0.25`, `_ilqr_v_stop_tol = 0.10`, builds time-indexed robust schedules via `_rebuild_acc_robust_schedules()`, and extends `self.x0` / `self.xg` to 6D.

### `set_init_pos(...)` (lines 55–64) & `set_goal_pos(...)` (lines 66–75)
Extend parent 4D pose setup by concatenating `[v, delta]`.

### `reset(self, rng: jax.Array)` (lines 78–88)
JIT-compiled. Returns 6D `State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(2), jnp.array(False), rng)`.

### `input_scaler(self, u_normalized)` (lines 91–95)
JIT-compiled. Maps `u_normalized` to `[u[0] * a_max, u[1] * omega_max]`.

### `tractor_trailer_dynamics(self, x, u)` (lines 98–124)
JIT-compiled nominal 6D ODE using `self.l1`, `self.l2`, `self.lh` (clips `v` and `delta` inside the position/heading rate equations).

### `tractor_trailer_dynamics_perturbed(self, x, u)` (lines 127–148)
JIT-compiled perturbed 6D ODE using time-invariant `self.l1_perturbed`, `self.l2_perturbed`, `self.lh_perturbed` (active in Level 2).

### `step_perturbed(self, state: State, action: jax.Array) -> State` (lines 151–186)
JIT-compiled 6D perturbed simulation step (Levels 0, 1, 2, 3):
- Splits `state.rng` into `(rng, rng_ctrl, rng_state)` so each step draws independent noise.
- Perturbs normalized control via `self.perturb_control(action_clipped, rng_ctrl)` and scales via `self.input_scaler`.
- Evaluates `self.tractor_trailer_dynamics_perturbed` and `self.perturb_state_dot`, integrating `q_final = q + self.dt * x_dot_pert`.
- Evaluates `check_obstacle_collision(q_final[:4], ..., use_perturbed=(self.perturbation_level == 2))`, `check_hitch_violation(q_final[:4])`, `get_reward(q_final[:4])`, and `get_preference_penalty(q_final, u_scaled)`.
- Returns updated `State` with `rng=rng`.

### `step(self, state: State, action: jax.Array) -> State` (lines 189–191) & `_step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State` (lines 194–281)
Nominal step functions used during diffusion planning, threading `rng_next` into `State`.

### Robust Tube Schedules & Two-Stage Backup (`_build_schedule_for_times`, `_rebuild_acc_robust_schedules`, `_init_robust_margins`, `_verify_ilqr_stage2`, `shielded_rollout`) (lines 288–521)
- **`_build_schedule_for_times` & `_rebuild_acc_robust_schedules`**: Resamples scalar or horizon-array `r_tube` and `b_hitch` onto `_rollout_dt` rollout steps (`_r_margin_rollout`, `_b_margin_rollout`) and `_ilqr_dt` iLQR steps (`_r_margin_ilqr`, `_b_margin_ilqr`).
- **`shielded_rollout(self, q, q_proposed)`** (lines 398–447):
  1. **Stage 1**: Runs fast single-step or multi-step braking rollout (`_check_safety_single_step` / `_check_safety_with_rollout` + `_rollout_step`) using tightened obstacle and hitch margins when `enable_robust_shielding=True`.
  2. **Stage 2** (opt-in via `enable_ilqr_backup=True`): Only when Stage 1 flags unsafe, invokes `jax.lax.cond(..., _verify_ilqr_stage2, ...)` calling `solve_ilqr_backup` ([mbd/robust/ilqr_backup.py](../robust/ilqr_backup.md)) and re-verifies the optimized recovery trajectory with exact tightened `check_obstacle_collision`, `check_hitch_violation`, and terminal stop check `|v[-1]| <= _ilqr_v_stop_tol`.
  3. Returns `q_proposed` if safe, or applies emergency braking/steering-centering step from `q` otherwise.

### `project_to_safe_set` (lines 393–412), `project_control_to_safe_set` (lines 415–429), `apply_guidance` (lines 432–449)
4D state slicing wrappers around parent projection and guidance methods.

### `get_preference_penalty` (lines 451–456), `get_reward` (lines 458–460), `get_terminal_reward` (lines 462–464), `action_size` (`2`, lines 467–468), `observation_size` (`6`, lines 471–472), `generate_demonstration_trajectory` (lines 474–486), `compute_demonstration_reward` (lines 488–493), `eval_xref_logpd` (lines 495–504)
4D slicing delegates to `TractorTrailer2d`.
