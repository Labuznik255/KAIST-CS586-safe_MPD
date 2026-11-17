# `mbd/robots/n_trailer2d.py`

**Source file:** [mbd/robots/n_trailer2d.py](../../../mbd/robots/n_trailer2d.py)

## Overview
Implements `NTrailer2d`, a general kinematic $N$-trailer model ($N \ge 2$) inheriting from `TractorTrailer2d`:
- **State ($(3+N)$D)**: $x = [p_x, p_y, \theta_0, \theta_1, \dots, \theta_N]$ where $\theta_0$ is the tractor heading and $\theta_1 \dots \theta_N$ are the $N$ trailer headings.
- **Control (2D)**: normalized $u_{\text{norm}} \in [-1, 1]^2$, scaled to $[v, \delta]$.
- Supports all 4 perturbation levels (Levels 0–3), including per-trailer time-invariant length uncertainty `self.trailer_lengths_perturbed` (shape `(N,)`) in Level 2.

---

## Class: `NTrailer2d(TractorTrailer2d)` (lines 11–627)

- **`__init__(self, *, num_trailers: int, trailer_lengths=None, trailer_front_offsets=None, trailer_rear_offsets=None, **kwargs)`** (lines 18–57): Initializes per-trailer geometry arrays (`trailer_lengths`, `trailer_lengths_nominal`, `trailer_lengths_perturbed`, `trailer_front_offsets`, `trailer_rear_offsets`), calls `_update_perturbed_trailer_lengths()`, and expands `x0` and `xg` to $(3+N)$D.
- **`_update_perturbed_trailer_lengths(self)`** (lines 59–76): Samples time-invariant Gaussian perturbations for all $N$ trailer lengths `self.trailer_lengths_perturbed` when `self.perturbation_level == 2` (or sets equal to `self.trailer_lengths_nominal` otherwise) and clears class-level JIT caches.
- **`set_perturbation(self, *args, **kwargs)`** (lines 78–81) & **`get_perturbation_info(self)`** (lines 83–88): Extends parent perturbation configuration and info dict with `trailer_lengths`.
- **`input_scaler(self, u_normalized)`** (lines 91–92).
- **`n_trailer_dynamics(self, x, u)`** (lines 95–133): Nominal $(3+N)$D ODE using `self.l1`, `self.lh`, and `self.trailer_lengths`.
- **`n_trailer_dynamics_perturbed(self, x, u)`** (lines 136–172) & **`tractor_trailer_dynamics_perturbed(self, x, u)`** (lines 175–177): Perturbed $(3+N)$D ODE using `self.l1_perturbed`, `self.lh_perturbed`, and `self.trailer_lengths_perturbed` (used by inherited `step_perturbed`).
- **`reset(self, rng: jax.Array)`** (lines 179–188): Returns $(3+N)$D `State(..., rng)`.
- **`set_init_pos(...)`** (lines 190–196) & **`set_goal_pos(...)`** (lines 198–203): Expand 4D poses to $(3+N)$D.
- **`step`** (lines 206–208), **`_step_internal`** (lines 211–254), **`_step_with_shielded_rollout`** (lines 257–259), **`_step_without_shielded_rollout`** (lines 262–265), **`shielded_rollout`** (lines 268–273): Nominal step and single-step shielding.
- **`get_preference_penalty`** (lines 276–294), **`apply_guidance`** (lines 297–311), **`get_last_trailer_position`** (lines 314–336), **`get_reward`** (lines 339–354), **`check_hitch_violation`** (lines 357–370).
- **`get_tractor_trailer_rectangles_all(self, x, use_perturbed: bool = False)`** (lines 373–417) & **`check_obstacle_collision(self, x, obs_circles, obs_rectangles, use_perturbed: bool = False)`** (lines 420–451): Multi-trailer bounding boxes and collision checks supporting `use_perturbed=True`.
- **`get_terminal_reward`** (lines 454–466), **`observation_size`** (`3 + self.num_trailers`, lines 469–470), **`action_size`** (`2`, lines 473–474), **`get_trailer_position`** (lines 476–478).
- **`setup_animation_patches`** (lines 481–529), **`render_rigid_body`** (lines 531–601), **`update_animation_patches`** (lines 603–627).
