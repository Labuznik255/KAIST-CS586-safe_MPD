# `mbd/robots/acc_tt2d.py`

**Source file:** [mbd/robots/acc_tt2d.py](../../../mbd/robots/acc_tt2d.py)

## Overview
Implements `AccTractorTrailer2d`, an acceleration- and steering-rate-controlled subclass of `TractorTrailer2d`:
- **State (6D)**: $x = [p_x, p_y, \theta_1, \theta_2, v, \delta]$
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[a, \omega] = [\dot{v}, \dot{\delta}]$ where $a \in [-a_{\max}, a_{\max}]$ and $\omega \in [-\omega_{\max}, \omega_{\max}]$.
- **Key Features**: Predictive **rollout-to-stop** safety verification inside `shielded_rollout` (`_check_safety_with_rollout` + `_rollout_step`), plus 6D perturbed dynamics (`tractor_trailer_dynamics_perturbed`) and `step_perturbed` for Levels 0–3 perturbations.

---

## Class: `AccTractorTrailer2d(TractorTrailer2d)` (lines 19–504)

### `__init__(self, **kwargs)` (lines 26–53)
Pops `a_max` (default `2.0` m/s²) and `omega_max` (default `1.0` rad/s), calls `super().__init__(**kwargs)` (which also initializes all perturbation attributes), pre-computes `_rollout_dt = 1.0`, `_v_threshold = 0.01`, `_max_rollout_steps`, and extends `self.x0` and `self.xg` from 4D to 6D with `[0.0, 0.0]`.

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

### `_step_with_shielded_rollout` (lines 284–287), `_step_without_shielded_rollout` (lines 290–296), `shielded_rollout` (lines 299–337), `_check_safety_single_step` (lines 340–345), `_rollout_step` (lines 348–368), `_check_safety_with_rollout` (lines 371–390)
Predictive rollout-to-stop safety shield for 6D acceleration dynamics. **Future Robust Shielding Hook**: Modify `_rollout_step` / `_check_safety_with_rollout` to account for disturbance tubes or worst-case parameter/control uncertainty.

### `project_to_safe_set` (lines 393–412), `project_control_to_safe_set` (lines 415–429), `apply_guidance` (lines 432–449)
4D state slicing wrappers around parent projection and guidance methods.

### `get_preference_penalty` (lines 451–456), `get_reward` (lines 458–460), `get_terminal_reward` (lines 462–464), `action_size` (`2`, lines 467–468), `observation_size` (`6`, lines 471–472), `generate_demonstration_trajectory` (lines 474–486), `compute_demonstration_reward` (lines 488–493), `eval_xref_logpd` (lines 495–504)
4D slicing delegates to `TractorTrailer2d`.
