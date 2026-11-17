# `mbd/robots/n_trailer2d.py`

**Source file:** [mbd/robots/n_trailer2d.py](../../../mbd/robots/n_trailer2d.py)

## Overview
Implements `NTrailer2d`, a general kinematic $N$-trailer model ($N \ge 2$) inheriting from `TractorTrailer2d`:
- **State ($(3+N)$D)**: $x = [p_x, p_y, \theta_0, \theta_1, \dots, \theta_N]$ where $\theta_0$ is the tractor heading and $\theta_1 \dots \theta_N$ are the $N$ trailer headings.
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[v, \delta]$.

---

## Class: `NTrailer2d(TractorTrailer2d)` (lines 11–538)

### `__init__(self, *, num_trailers: int, trailer_lengths=None, trailer_front_offsets=None, trailer_rear_offsets=None, **kwargs)` (lines 18–55)
1. Asserts `num_trailers >= 2` and calls `super().__init__(num_trailers=num_trailers, **kwargs)`.
2. Creates length-$N$ JAX arrays `trailer_lengths`, `trailer_front_offsets`, and `trailer_rear_offsets` (defaults: all lengths `l2`, front offsets `lf2`, rear offsets `lr` for trailers $1 \dots N-1$ and `lr2` for the last trailer $N$).
3. Expands 4D `self.x0` and `self.xg` to $(3+N)$D via `_augment_state` by repeating the base trailer angle across all $N$ trailers.
4. Initializes animation patch lists (`trailer_bodies`, `trailer_wheels`, `trailer_hitch_lines_front`, `trailer_hitch_lines_back`).

### `input_scaler(self, u_normalized)` (lines 57–59)
JIT-compiled wrapper calling `super().input_scaler(u_normalized)`.

### `n_trailer_dynamics(self, x, u)` (lines 61–100)
JIT-compiled $(3+N)$D kinematic $N$-trailer ODE:
- Computes tractor rates $\dot{p}_x, \dot{p}_y, \dot{\theta}_0$.
- Uses `jax.lax.fori_loop(0, self.num_trailers, loop, init)` to propagate axle speed and angular rate $\dot{\theta}_i$ sequentially from trailer $0$ through trailer $N-1$ using hitch length `self.lh` and per-trailer lengths `self.trailer_lengths[i]`.
- **Perturbation hook**: Add kinematic disturbances, hitch slip, or trailer-specific parameter perturbations inside `loop` or on `x_dot` before returning.

### `reset(self, rng: jax.Array)`
Returns `State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(2), jnp.array(False), rng)`.

### `_update_perturbed_trailer_lengths()`, `n_trailer_dynamics_perturbed(self, x, u)`, `tractor_trailer_dynamics_perturbed(self, x, u)`
Samples time-invariant perturbed trailer lengths `self.trailer_lengths_perturbed` (size $N$) for Level 2 parameter uncertainty and evaluates the $(3+N)$D $N$-trailer ODE using `self.l1_perturbed`, `self.lh_perturbed`, and `self.trailer_lengths_perturbed`.

### `set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0)` (lines 105–111)
Calls `super().set_init_pos(...)` and expands `self.x0` to $(3+N)$D by repeating `theta2` across all $N$ trailers.

### `set_goal_pos(self, x=None, y=None, theta1=None, theta2=None)` (lines 113–118)
Calls `super().set_goal_pos(...)` and expands `self.xg` to $(3+N)$D by repeating `theta2` across all $N$ trailers.

### `step(self, state: State, action: jax.Array) -> State` (lines 120–123)
JIT-compiled wrapper calling `_step_internal(state, action, self.visualization_mode)`.

### `_step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State` (lines 125–168)
JIT-compiled (`visualization_mode` static at arg 3).
- Asserts `q.shape[0] == 3 + self.num_trailers`.
- Raises `AssertionError("Projection not implemented for NTrailer2d")` if `self.enable_projection` is True.
- Otherwise integrates `euler(self.n_trailer_dynamics, q, u_scaled, self.dt)` and runs guidance, shielded rollout, or naive penalty mode, returning updated `State`.

### `_step_with_shielded_rollout(self, data)` (lines 170–173)
JIT-compiled helper calling `self.shielded_rollout(q, q_proposed)`.

### `_step_without_shielded_rollout(self, q_proposed)` (lines 175–179)
JIT-compiled helper checking obstacle collision and hitch violations on `q_proposed`.

### `shielded_rollout(self, q, q_proposed)` (lines 181–187)
JIT-compiled 1-step shield reverting to `q` on enabled obstacle collision or any consecutive hitch angle violation.

### `get_preference_penalty(self, x_or_u, u=None)` (lines 189–208)
JIT-compiled motion-preference penalty matching `TractorTrailer2d.get_preference_penalty`.

### `apply_guidance(self, q_proposed, step_size=0.05, max_steps=3)` (lines 210–225)
JIT-compiled (`max_steps` static at arg 3). Runs gradient descent on `_guidance_function` (inherited from `TractorTrailer2d`—note that parent's `_guidance_function` unpacks `x[:4]` only) and clips `x_new[2:]` to `[-2*pi, 2*pi]`.

### `get_last_trailer_position(self, x)` (lines 227–250)
JIT-compiled `fori_loop` walking forward-to-backward through all $N$ hitches and trailer centers to return the last trailer's axle center `[last_cx, last_cy]`.

### `get_reward(self, q)` (lines 252–268)
JIT-compiled simplified position-only reward tracking tractor front axle (forward preference), last trailer axle (backward preference), or `min(d_tr, d_la)` (no preference).

### `check_hitch_violation(self, x)` (lines 270–284)
JIT-compiled via `jax.lax.scan`. Checks $|\text{wrap\_pi}(\theta_0 - \theta_1)| > \phi_{\max}$ and every consecutive pair $|\text{wrap\_pi}(\theta_{k-1} - \theta_k)| > \phi_{\max}$.

### `get_tractor_trailer_rectangles_all(self, x)` (lines 286–328)
JIT-compiled. Computes `tractor_corners` `(4, 2)` and `corners_out` `(N, 4, 2)` for all $N$ trailers iteratively via `jax.lax.fori_loop`.

### `check_obstacle_collision(self, x, obs_circles, obs_rectangles)` (lines 330–362)
JIT-compiled. Checks tractor and all $N$ trailer rectangles (`vmap`ped) against all `obs_circles` and `obs_rectangles`.

### `get_terminal_reward(self, q)` (lines 364–377)
JIT-compiled terminal position reward using tractor or last-trailer distance.

### `observation_size` (`@property`, lines 379–381) → `3 + self.num_trailers`
### `action_size` (`@property`, lines 383–385) → `2`

### `get_trailer_position(self, x)` (lines 386–389)
JIT-compiled alias to `get_last_trailer_position(x)`.

### `setup_animation_patches(self, ax)` (lines 392–440)
Creates Matplotlib patches for the tractor, $N$ trailers, wheels, tractor hitch line, $N$ front hitch lines, and $N-1$ back hitch lines.

### `render_rigid_body(self, x, u=None)` (lines 442–512)
Computes `Affine2D` transforms for tractor and all $N$ trailers plus hitch line segments.

### `update_animation_patches(self, x, u=None)` (lines 514–538)
Applies transforms and hitch line coordinates from `render_rigid_body` to all Matplotlib artists.

---

## Extension Notes
- **Perturbations**: `n_trailer_dynamics` (lines 61–100) propagates kinematics through `jax.lax.fori_loop`; adding per-trailer hitch slip, parameter mismatch in `self.trailer_lengths[i]`, or additive rate disturbances here propagates naturally down the trailer chain.
- **Note on `tests/stat_mbd.py` & `tests/test_planners/test_base.py`**: Those evaluation scripts slice `state[:4]` when calling collision/hitch checks, whereas `NTrailer2d` needs the full `(3+N)`D state (as handled in `mbd/utils.py` line 402–403).
