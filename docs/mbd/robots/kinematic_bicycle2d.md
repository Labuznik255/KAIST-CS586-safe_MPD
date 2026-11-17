# `mbd/robots/kinematic_bicycle2d.py`

**Source file:** [mbd/robots/kinematic_bicycle2d.py](../../../mbd/robots/kinematic_bicycle2d.py)

## Overview
Implements `KinematicBicycle2d`, a single-body kinematic bicycle model (no trailer) inheriting from `TractorTrailer2d`:
- **State (3D)**: $x = [p_x, p_y, \theta_1]$
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[v, \delta]$.

---

## Class: `KinematicBicycle2d(TractorTrailer2d)` (lines 20–841)

### `__init__(self, **kwargs)` (lines 30–38)
Calls `super().__init__(**kwargs)` and truncates `self.x0` and `self.xg` to 3D (`[:3]`).

### `set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0)` (lines 40–56)
Sets 3D `self.x0 = [px, py, theta1]`. Uses parent parking geometry if `case == "parking"` and `dx, dy` are given, then slices `[:3]`; `theta2` is ignored.

### `set_goal_pos(self, x=None, y=None, theta1=None, theta2=None)` (lines 58–75)
Updates 3D `self.xg = [x, y, theta1]`; `theta2` is ignored.

### `reset(self, rng: jax.Array)`
JIT-compiled. Returns 3D `State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(2), jnp.array(False), rng)`.

### `bicycle_dynamics_perturbed(self, x, u)` & `tractor_trailer_dynamics_perturbed(self, x, u)`
JIT-compiled 3D kinematic bicycle ODE using `self.l1_perturbed` (used by inherited `step_perturbed` across Levels 0, 1, 2, 3).

### `bicycle_dynamics(self, x, u)` (lines 82–97)
JIT-compiled 3D kinematic bicycle ODE: $\dot{p}_x = v\cos\theta_1$, $\dot{p}_y = v\sin\theta_1$, $\dot{\theta}_1 = \frac{v}{l_1}\tan\delta$.

### `tractor_trailer_dynamics(self, x, u)` (lines 99–102)
JIT-compiled alias calling `self.bicycle_dynamics(x, u)`.

### `get_tractor_rectangle(self, x)` (lines 104–132)
JIT-compiled. Computes tractor bounding box corners, center, and angle from 3D state `x`.

### `check_obstacle_collision(self, x, obs_circles, obs_rectangles)` (lines 134–191)
JIT-compiled. Checks tractor-only rectangle against `obs_circles` and `obs_rectangles` using parent `check_rectangle_circle_collision` and `check_rectangle_rectangle_collision`.

### `check_hitch_violation(self, x)` (lines 193–196)
JIT-compiled. Always returns `False` (no trailer).

### `check_collision(self, x, obs_circles, obs_rectangles)` (lines 198–201)
JIT-compiled wrapper calling `check_obstacle_collision`.

### `_step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State` (lines 203–259)
JIT-compiled (`visualization_mode` static at arg 3). Mirrors `TractorTrailer2d._step_internal` using `euler(self.tractor_trailer_dynamics, ...)` for projection, guidance, shielded rollout, and naive penalty modes.

### `get_trailer_back_position(self, x)` (lines 261–269)
JIT-compiled fallback returning tractor rear overhang point `[px - lh*cos(theta1), py - lh*sin(theta1)]` so evaluation scripts (`stat_mbd.py`, `test_base.py`) work uniformly.

### `get_reward(self, q)` (lines 272–316)
JIT-compiled stage reward for 3D state: blends `r_pos` (tractor front axle distance to `xg[:2]`) and `r_hdg` ($\theta_1$ alignment with `xg[2]`) via logistic distance switch `w_theta`; no articulation term.

### `get_terminal_reward(self, q)` (lines 318–329)
JIT-compiled terminal position reward for tractor front axle.

### `observation_size` (`@property`, lines 331–333) → `3`

### `generate_demonstration_trajectory(...)` (lines 335–361)
Temporarily pads `self.xg` to 4D, calls `super().generate_demonstration_trajectory(...)`, slices `self.xref = xref_4d[:, :3]`, and restores `self.xg`.

### `eval_xref_logpd(self, xs, motion_preference=None)` (lines 363–416)
Evaluates reference trajectory log-probability density over 3D states (position + $\theta_1$, no $\theta_2$) via nested `@jax.jit def _eval_xref_logpd`.

### `render(self, ax, xs: jnp.ndarray)` (lines 418–501)
Draws parking spaces, circular/rectangular obstacles, and bicycle trajectory `xs[:, :2]`.

### `setup_animation_patches(self, ax, tractor_color='blue')` (lines 503–527)
Creates `tractor_body` and 2 `tractor_wheels` Matplotlib patches.

### `get_bicycle_positions(self, x)` (lines 529–545)
NumPy helper returning `tractor_rear`, `tractor_front`, and `theta1`.

### `render_rigid_body(self, x, u=None)` (lines 547–581)
Returns `Affine2D` transforms for tractor body and front/rear wheels.

### `update_animation_patches(self, x, u=None)` (lines 583–594)
Applies transforms from `render_rigid_body` to body and wheel patches.

### `_numpy_projection_function(self, args_tuple)` (lines 596–771)
SciPy `trust-constr` projection solver for 3D bicycle state (tractor-only obstacle SDF constraints, no hitch constraint).
- **Nested helpers**: `bicycle_dynamics_np`, `euler_np` (RK4), `input_scaler_np`, `get_tractor_rectangle_np`, `_sdf_oriented_box_np`, `_sdf_two_boxes_np`, `_signed_dist_rect_circle_np`, `_signed_dist_rect_rect_np`, `constraint_function_np`, `objective`, `constraint_function`.

### `_guidance_function(self, x)` (lines 773–800)
JIT-compiled tractor-only circular-obstacle barrier cost.

### `apply_guidance(self, q_proposed, step_size=0.05, max_steps=3)` (lines 802–841)
JIT-compiled (`max_steps` static at arg 3). Runs `max_steps` of gradient descent on `_guidance_function` with 3D state clipping.

---

## Extension Notes
- Useful reference when overriding `TractorTrailer2d` for a **different state dimension** (here 3D instead of 4D) or simplified/changed geometry & collision checks.
- Note that `bicycle_dynamics` (lines 82–97) is the single-body dynamics entry point if you ever want a dynamic single-vehicle bicycle model (without trailers).
