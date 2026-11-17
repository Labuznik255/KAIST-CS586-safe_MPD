# `mbd/planners/mbd_planner.py`

**Source file:** [mbd/planners/mbd_planner.py](../../../mbd/planners/mbd_planner.py)

## Overview
Implements the core **Safe Model Predictive Diffusion (Safe-MPD)** planner: configuration (`MBDConfig`), JIT function caching, reverse diffusion sampling (`run_diffusion` / `reverse_once`), steering smoothness reward, post-processing trajectory rollout (including guided vs. unguided comparison), and result saving/rendering.

---

## Global Utilities & Cache

### `count_compilation(f)` (lines 28–35)
Decorator incrementing global `compilation_count` and logging debug messages whenever a decorated function is compiled/invoked.

### `clear_jit_cache()` (lines 41–47)
Clears `_jit_function_cache` (`dict`). Must be called between runs whenever static parameters, environment obstacles, or vehicle geometry/perturbation parameters change so `step_env_jit`, `rollout_us_jit`, `rollout_us_with_terminal_jit`, and `reverse_once` recompile against the updated `env`.

---

## Configuration

### `@dataclass class MBDConfig` (lines 50–118)
Stores all planner, safety, vehicle geometry, reward, animation, and scalability parameters:
- **Experiment/Env**: `seed`, `env_name` (`"acc_tt2d"`, `"tt2d"`, `"n_trailer2d"`, `"kinematic_bicycle2d"`), `case` (`"parking"` or `"navigation"`), `verbose`, `dt`, `num_trailers`.
- **Diffusion**: `Nsample` (20000), `Hsample` (50), `Ndiffuse` (100), `temp_sample` (1e-5), `beta0` (1e-5), `betaT` (1e-2), `enable_demo`.
- **Motion & Safety**: `motion_preference` (`0`, `1`, `-1`, `2`, `-2`), `collision_penalty`, `hitch_penalty`, `enable_shielded_rollout_collision`, `enable_shielded_rollout_hitch`, `enable_projection`, `enable_guidance`.
- **Physical Geometry & Limits**: `l1`, `l2`, `lh`, `lf1`, `lr`, `lf2`, `lr2`, `tractor_width`, `trailer_width`, `v_max`, `delta_max_deg`, `a_max`, `omega_max`.
- **Reward Shaping**: `reward_threshold`, `ref_reward_threshold`, `max_w_theta`, `hitch_angle_weight`, `terminal_reward_threshold`, `terminal_reward_weight`, `d_thr_factor`, `k_switch`, `steering_weight`, `preference_penalty_weight`, `heading_reward_weight`, `ref_pos_weight`, `ref_theta1_weight`, `ref_theta2_weight`.
- **Rendering**: `render`, `save_animation`, `show_animation`, `save_denoising_animation`, `frame_skip`.

### `dict_to_config_obj(config_dict)` (lines 120–186)
Converts a Python `dict` into an `MBDConfig` dataclass with explicit type casting (`int`, `float`, `bool`, `str`). Note: does not currently read `num_trailers` or `frame_skip` from `config_dict`.

---

## Core Planner

### `run_diffusion(args=None, env=None)` (lines 189–751)
Runs the full Safe-MPD planning loop on `env` using `args` (`MBDConfig` or `dict`).
- **Returns**: `(rew_final, Y0, trajectory_states, timing_info)`
  - `rew_final`: Scalar total reward of final `Y0`.
  - `Y0`: Final denoised (and shielded/applied) control sequence, shape `(Hsample, Nu)`.
  - `trajectory_states`: Rolled-out state sequence, shape `(Hsample + 1, Nx)`.
  - `timing_info`: Breakdown dict (`total_time`, `demo_generation_time`, `compilation_time`, `jit_setup_time`, `warmup_time`, `reward_compile_time`, `diffusion_setup_time`, `pure_diffusion_time`, `post_processing_time`, `overhead_time`).

#### Execution Stages & Nested Functions inside `run_diffusion`
1. **Setup & Demo (lines 203–228)**: Converts dict args if needed; if `args.enable_demo`, calls `env.generate_demonstration_trajectory(...)` and `env.compute_demonstration_reward()`.
2. **JIT Setup & Cache (lines 230–252)**: Looks up or creates `step_env_jit = jax.jit(env.step)`, `rollout_us_jit`, and `rollout_us_with_terminal_jit` keyed by `f"{type(env).__name__}_env_funcs"`, and calls `state_init = env.reset(rng_reset)`.
3. **Warmup (lines 257–294)**: Compiles `step_env_jit`, `rollout_us_jit`, and `rollout_us_with_terminal_jit` with dummy zero controls of shape `(Hsample, Nu)`.
4. **Diffusion Noise Schedule (lines 298–301)**: Linear `betas` from `beta0` to `betaT`, `alphas = 1 - betas`, `alphas_bar = cumprod(alphas)`, `sigmas = sqrt(1 - alphas_bar)`.
5. **Nested `@jax.jit def compute_steering_reward(Y0s)` (lines 305–336)**:
   - Takes `(Nsample, Hsample, Nu)` controls and extracts `deltas = Y0s[:, :, 1]` (normalized steering / steering rate).
   - Computes magnitude reward `1 - |delta|^2` and rate reward `1 - clip(|diff(deltas * env.delta_max)| / (0.1 * env.delta_max), 0, 1)^2` (padded with `1.0` at $t=0$), averages them (`0.5 * (r_delta + r_deltadot_padded)`), and takes mean over horizon.
6. **Nested `@partial(jax.jit, static_argnums=(2,)) def reverse_once(carry, state_init, config_params_tuple)` (lines 346–428)**:
   - Cached in `_jit_function_cache["reverse_once_function"]`.
   - Samples `Nsample` noisy control trajectories `Y0s = eps_u * sigmas[i]/sqrt(alphas_bar[i-1]) + Ybar_i` and clips to `[-1, 1]`.
   - Enforces strict forward/backward velocity bounds on `Y0s[:, :, 0]` if `motion_preference == 2` (`[0, 1]`) or `-2` (`[-1, 0]`).
   - Vectorizes rollout across all `Nsample` trajectories via `jax.vmap(rollout_us_with_terminal_jit, in_axes=(None, 0))(state_init, Y0s)` -> `(rewss, qs, us_applied_seq)`.
   - Replaces sampled controls with actually applied (shielded/projected) controls `Y0s_effective = us_applied_seq`.
   - Blends `rews + env.steering_weight * compute_steering_reward(Y0s_effective)`.
   - Standardizes rewards `(rews_combined - mean) / std / temp_sample`; if `enable_demo`, evaluates `env.eval_xref_logpd(q, motion_preference)` and takes elementwise `max(logp0, logpdemo)`.
   - Computes softmax weights and weighted average `Ybar = einsum("n,nij->ij", weights, Y0s_effective)`, then performs DDIM/score reverse step to obtain `Ybar_im1`.
7. **Nested `def reverse(YN, rng, state_init)` (lines 478–507)**:
   - Iterates `i = Ndiffuse - 1` down to `1` calling `reverse_once`, storing intermediate `Yi` in `Ybars` of shape `(Ndiffuse - 1, Hsample, Nu)`.
   - Afterwards (lines 513–514), runs `rollout_us_with_terminal_jit(state_init, Y0)` once more so `Y0` is replaced by the final shielded `Y0_applied`.
8. **Post-Processing Rollout & Perturbed Simulation Rollout**:
   - Sequentially rolls out `Y0` from `state_init` into `nominal_trajectory_states` (`trajectory_states`) of shape `(Hsample + 1, Nx)`.
   - Rolls out `Y0` in the perturbed environment via `rollout_perturbed_us(jax.jit(env.step_perturbed), state_pert_init, Y0)` to produce `perturbed_trajectory_states` and `perturbed_actions`.
   - If `args.enable_guidance` and `hasattr(env, 'visualization_mode')`, defines `step_with_guided_mode` calling `env._step_internal(state, action, True)` for the guided visualization trajectory, and also clears the JIT cache and runs `step_without_guidance` (`env._step_internal(state, action, False)`) to produce `unguided_trajectory_states` for comparison.
9. **Rendering & Saving**:
   - Saves `mu_0ts.npy`, `final_actions.npy`, `perturbed_actions.npy`, `nominal_trajectory_states.npy`, `perturbed_trajectory_states.npy`, and `trajectory_states.npy` under `results/{args.env_name}/`.
   - When `args.render` is True: plots nominal trajectory and overlays perturbed trajectory in `rollout.png` and `rollout.svg`, and runs `create_animation(...)` for nominal trajectory (`nominal_trajectory_animation.mp4`) and then perturbed trajectory (`perturbed_trajectory_animation.mp4`).

---

## Standalone Entry Point (`__main__`, lines 756–833)
Creates default `MBDConfig()`, builds `env` via `mbd.envs.get_env(...)`, sets initial/goal poses via `env.set_init_pos(...)` and `env.set_goal_pos(...)`, and calls `run_diffusion(args=config, env=env)`.

---

## Extension Points
- **New vehicle class**:
  1. Add any new physical/model parameters to `MBDConfig` (lines 50–118), `dict_to_config_obj` (lines 120–186), and the `mbd.envs.get_env(...)` call in `__main__` (lines 773–817).
  2. Check `compute_steering_reward` (line 312: `Y0s[:, :, 1]`, line 315: `env.delta_max`) and `reverse_once` velocity clipping (line 371: `Y0s[:, :, 0]`). If your new vehicle has `Nu != 2` or different action semantics, generalize or dispatch these via methods on `env`.
  3. Notice `_jit_function_cache` (lines 236, 254): `"reverse_once_function"` is cached globally under a single key, so if you switch vehicle types in the same Python process, call `clear_jit_cache()`.
- **Perturbations on existing tractor-trailer models**:
  1. To evaluate **model mismatch** (plan with nominal dynamics, execute on perturbed dynamics), replace or wrap `step_env_jit_final_rollout` in post-processing (lines 550–565) with a perturbed environment step, while keeping `reverse_once` using the nominal `env`.
  2. To inject **stochastic perturbations inside planning rollouts**, notice `reverse_once` (line 395) calls `jax.vmap(rollout_us_with_terminal_jit, in_axes=(None, 0))(state_init, Y0s)`—passing a single `state_init` broadcast across `Nsample`. If per-sample noise is desired, pass batched per-sample `rng` keys or include `rng` in `State` and split it per sample.
