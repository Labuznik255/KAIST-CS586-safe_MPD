# `mbd/utils.py`

**Source file:** [mbd/utils.py](../../mbd/utils.py)

## Overview
Provides JAX-accelerated trajectory rollout primitives (`eval_us`, `rollout_us`, `rollout_us_with_terminal`, `rollout_perturbed_us`), environment metadata persistence (`save_env_metadata`, `load_env_from_results`), and Matplotlib + FFmpeg animation and video export utilities (`setup_animation_saving`, `export_video`, `merge_progress_videos`, `create_animation`, `create_denoising_animation`).

---

## Trajectory Rollout & Metadata Functions

### `eval_us(step_env, state, us)` (lines 19–25)
Evaluates a control sequence `us` of shape `(H, Nu)` starting from `state` using `jax.lax.scan`.
- **Returns**: `rew_seq` of shape `(H,)`.

### `rollout_us(step_env, state, us)` (lines 27–35)
Rolls out control sequence `us` of shape `(H, Nu)` from `state` using `jax.lax.scan`, returning `(rew_seq, pipeline_state_seq, applied_action_seq)`. Performs a warm-init step `state_init = step_env(state, zero_u)` prior to `jax.lax.scan`.

### `rollout_us_with_terminal(step_env, env, state, us)` (lines 37–52)
Same scan structure as `rollout_us`, then computes:
$$\text{total\_reward} = \text{mean}(\text{rew\_seq}) + \text{env.terminal\_reward\_weight} \cdot \text{env.get\_terminal\_reward}(\text{final\_state.pipeline\_state})$$
- **Returns**: `(total_reward, pipeline_state_seq, applied_action_seq)`.

### `rollout_perturbed_us(step_perturbed_env, state, us)` (lines 55–63)
Rolls out nominal control sequence `us` of shape `(H, Nu)` through `step_perturbed_env` (`env.step_perturbed`) using `jax.lax.scan`, prepending `state.pipeline_state[None, :]` to form `full_states` of shape `(H + 1, Nx)`.
- **Returns**: `(rew_seq, full_states, applied_action_seq)`.

### `save_env_metadata(env, args, results_path)` (lines 66–111)
Serializes environment configuration (`env_name`, `case`, `dt`, `H`, `num_trailers`, `motion_preference`, `x0`, `xg`, `obs_circles`, `obs_rectangles`, `x_range`, `y_range`, geometry parameters, `parking_config`, and `perturbation_info`) to `{results_path}/env_metadata.json` so standalone scripts (`tests/test_perturbations.py`, `render.py`) can reconstruct the exact environment.

### `load_env_from_results(results_dir, env_name_override=None, case_override=None)` (lines 114–233)
Reconstructs `(env, config, meta)` from `results_dir`. Reads `env_metadata.json` if present; otherwise infers `env_name`, `num_trailers`, `H`, `x0`, and `xg` from `nominal_trajectory_states.npy` / `trajectory_states.npy` and `final_actions.npy`.

---

## Animation & Video Utilities

### `setup_animation_saving(env_name, animation_type="trajectory")` (lines 248–264)
Creates `results/{env_name}/animations/{animation_type}` and deletes any existing `*.png` frames inside it. Returns `type_path`.

### `export_video(env_name, animation_type="trajectory", video_name=None)` (lines 267–305)
Invokes `ffmpeg` via `subprocess.call` to encode `frame_%04d.png` into an MP4 video (`10` fps input for `"trajectory"`, `30` fps for `"denoising"`, scaled to `1920x1080` at `30` fps `yuv420p`) and removes the PNG frames on success.

### `merge_progress_videos(env_name, animation_type="trajectory", prefix="progress_trial_", output_name="progress_merged.mp4")` (lines 308–350)
Concatenates per-trial `{prefix}*.mp4` videos into a single MP4 via `ffmpeg -f concat -c copy`.

### `create_animation(env, trajectory_states, trajectory_actions, args, guided_trajectory_overlay=None, progress_success_positions=None, progress_fail_positions=None, video_name=None, nominal_trajectory_overlay=None, trajectory_label=None)` (lines 352–664)
Renders an animated Matplotlib visualization of the trajectory frame-by-frame:
- Draws parking lot spaces (if `args.case == "parking"`), circular/rectangular obstacles, start/goal markers, optional cumulative success/fail scatter overlays, optional `guided_trajectory_overlay`, and optional `nominal_trajectory_overlay` (used when animating a perturbed trajectory against the nominal trajectory).
- Calls `env.setup_animation_patches(ax)` and, on each frame, `env.update_animation_patches(state_np, action_np)`.
- Checks `env.check_obstacle_collision` and `env.check_hitch_violation` at each frame and renders `'!'` markers on collision (`#FF3C3C`) or jackknifing (`#CC00FF`).
- Exports video via `export_video(...)` when `args.save_animation` is True.

### `create_denoising_animation(env, Yi, args, step_env_jit, state_init, frame_skip=1)` (lines 667–837)
Visualizes how the trajectory evolves over diffusion denoising steps `Yi` (shape `(Ndiffuse-1, Hsample, Nu)`) and exports `denoising_process_animation.mp4`.
