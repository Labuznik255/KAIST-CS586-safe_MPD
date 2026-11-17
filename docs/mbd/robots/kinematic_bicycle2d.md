# `mbd/robots/kinematic_bicycle2d.py`

**Source file:** [mbd/robots/kinematic_bicycle2d.py](../../../mbd/robots/kinematic_bicycle2d.py)

## Overview
Implements `KinematicBicycle2d`, a single-body kinematic bicycle model (no trailer) inheriting from `TractorTrailer2d`:
- **State (3D)**: $x = [p_x, p_y, \theta_1]$
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[v, \delta]$.
- Supports all 4 perturbation levels (Levels 0–3) via `bicycle_dynamics_perturbed`, `tractor_trailer_dynamics_perturbed`, `use_perturbed` collision geometry, and inherited `step_perturbed`.

---

## Class: `KinematicBicycle2d(TractorTrailer2d)` (lines 20–870)

- **`__init__(self, **kwargs)`** (lines 30–38): Calls `super().__init__(**kwargs)` and truncates `self.x0` and `self.xg` to 3D (`[:3]`).
- **`set_init_pos(...)`** (lines 40–56) & **`set_goal_pos(...)`** (lines 58–75): Set 3D `[px, py, theta1]`.
- **`reset(self, rng: jax.Array)`** (lines 78–88): Returns 3D `State(..., rng)`.
- **`bicycle_dynamics(self, x, u)`** (lines 91–105) & **`tractor_trailer_dynamics(self, x, u)`** (lines 122–124): Nominal 3D bicycle ODE using `self.l1`.
- **`bicycle_dynamics_perturbed(self, x, u)`** (lines 108–119) & **`tractor_trailer_dynamics_perturbed(self, x, u)`** (lines 127–129): Perturbed 3D bicycle ODE using time-invariant `self.l1_perturbed` (used by inherited `step_perturbed`).
- **`get_tractor_rectangle(self, x, use_perturbed: bool = False)`** (lines 132–160) & **`check_obstacle_collision(self, x, obs_circles, obs_rectangles, use_perturbed: bool = False)`** (lines 163–219): Tractor-only bounding box and collision checks using `self.l1_perturbed` when `use_perturbed=True`.
- **`check_hitch_violation(self, x)`** (lines 222–224): Always returns `False`.
- **`check_collision(self, x, obs_circles, obs_rectangles)`** (lines 227–229).
- **`_step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State`** (lines 232–288): Nominal 3D step threading `rng_next` into `State`.
- **`get_trailer_back_position(self, x)`** (lines 291–298), **`get_reward(self, q)`** (lines 302–345), **`get_terminal_reward(self, q)`** (lines 348–358), **`observation_size`** (`3`, lines 361–362).
- **`generate_demonstration_trajectory`** (lines 364–390), **`eval_xref_logpd`** (lines 392–445).
- **`render`** (lines 447–530), **`setup_animation_patches`** (lines 532–554), **`get_bicycle_positions`** (lines 558–574), **`render_rigid_body`** (lines 576–610), **`update_animation_patches`** (lines 612–621).
- **`_numpy_projection_function`** (lines 625–800), **`_guidance_function`** (lines 803–829), **`apply_guidance`** (lines 832–870).
