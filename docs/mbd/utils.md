# `mbd/utils.py`

**Source file:** [mbd/utils.py](../../mbd/utils.py)

## Overview
Provides JAX-accelerated trajectory evaluation/rollout primitives (`eval_us`, `rollout_us`, `rollout_us_with_terminal`) and Matplotlib + FFmpeg animation and video export utilities (`setup_animation_saving`, `export_video`, `merge_progress_videos`, `create_animation`, `create_denoising_animation`).

---

## Trajectory Rollout Functions

### `eval_us(step_env, state, us)` (lines 18–24)
Evaluates a control sequence `us` of shape `(H, Nu)` starting from `state` using `jax.lax.scan`.
- **Nested `step(state, u)`**: calls `state = step_env(state, u)` and returns `(state, state.reward)`.
- **Returns**: `rew_seq` of shape `(H,)`.

### `rollout_us(step_env, state, us)` (lines 26–34)
Rolls out control sequence `us` of shape `(H, Nu)` from `state` using `jax.lax.scan`, returning rewards, states, and actually applied controls (which may differ from `us` when shielding/projection modifies actions).
- **Important behavior (lines 31–32)**: Performs a warm-initialize step `state_init = step_env(state, zero_u)` **before** scanning `us`, so the carry PyTree structure/dtypes match `step_env`'s output.
- **Returns**: `(rew_seq, pipeline_state_seq, applied_action_seq)`.

### `rollout_us_with_terminal(step_env, env, state, us)` (lines 36–51)
Same scan structure as `rollout_us` (including the `zero_u` warm-init step on line 41–42), then computes the planning objective:
$$\text{total\_reward} = \text{mean}(\text{rew\_seq}) + \text{env.terminal\_reward\_weight} \cdot \text{env.get\_terminal\_reward}(\text{final\_state.pipeline\_state})$$
- **Returns**: `(total_reward, pipeline_state_seq, applied_action_seq)`.

> **Extension Gotcha (Dynamic Vehicles & Perturbations)**:
> Notice lines 31–32 and 41–42 apply `zero_u` for one extra `step_env` call before `jax.lax.scan(step, state_init, us)`. For kinematic models at rest (`v=0`), `u=0` is a no-op; for dynamic models with nonzero initial velocity or active perturbations/drift at `u=0`, this advances the initial state by `dt` before the first control in `us` is applied! Also line 31/41 falls back to `jnp.zeros((2,))` if `us.shape[0] == 0`, assuming `Nu=2`.

---

## Animation & Video Utilities

### `setup_animation_saving(env_name, animation_type="trajectory")` (lines 66–82)
Creates `results/{env_name}/animations/{animation_type}` relative to `mbd.__path__[0]` and deletes any existing `*.png` frames inside it. Returns `type_path`.

### `export_video(env_name, animation_type="trajectory", video_name=None)` (lines 85–123)
Invokes `ffmpeg` via `subprocess.call` to encode `frame_%04d.png` into an MP4 video (`10` fps input for `"trajectory"`, `30` fps for `"denoising"`, scaled to `1920x1080` at `30` fps `yuv420p`) and removes the PNG frames on success.

### `merge_progress_videos(env_name, animation_type="trajectory", prefix="progress_trial_", output_name="progress_merged.mp4")` (lines 126–168)
Finds all `{prefix}*.mp4` files in sorted order, writes `video_list.txt`, concatenates them with `ffmpeg -f concat -c copy`, and deletes `video_list.txt`. Used by `tests/stat_mbd.py`.

### `create_animation(env, trajectory_states, trajectory_actions, args, guided_trajectory_overlay=None, progress_success_positions=None, progress_fail_positions=None, video_name=None)` (lines 170–472)
Renders an animated Matplotlib visualization of the planned trajectory frame-by-frame:
- Draws parking lot spaces (if `args.case == "parking"`), circular/rectangular obstacles, start/goal markers, optional cumulative success/fail scatter overlays, and optional `guided_trajectory_overlay`.
- Calls `env.setup_animation_patches(ax)` and, on each frame, `env.update_animation_patches(state_np, action_np)`.
- **Nested class `SquareHandler(HandlerPatch)` (lines 329–342)**: custom legend patch handler keeping collision (`#FF3C3C`) and jackknifing (`#CC00FF`) markers square.
- **State slicing for violation markers (lines 398–415)**:
  - If `args.env_name == "kinematic_bicycle2d"`: passes `state_np[:3]` to `env.check_obstacle_collision` and skips `check_hitch_violation`.
  - Else if `getattr(env, 'num_trailers', 1) >= 2`: passes full `state_np`.
  - Else: passes `state_np[:4]` (assumes first 4 elements are `[px, py, theta1, theta2]`).
  - Draws `'!'` exclamation markers next to the vehicle when a collision or hitch violation occurs.
- Saves frames at 100 dpi (or 300 dpi if progress overlay is enabled) and calls `export_video(...)` when `args.save_animation` is True.

> **Extension Note (New Vehicle Class)**:
> Your new vehicle class must implement `setup_animation_patches(ax)` and `update_animation_patches(state_np, action_np)` and keep `[px, py, theta1, theta2]` in `state[:4]` (or update lines 398–415 in `create_animation` if collision/hitch checking requires higher-dimensional state).

### `create_denoising_animation(env, Yi, args, step_env_jit, state_init, frame_skip=1)` (lines 475–645)
Visualizes how the trajectory evolves over diffusion denoising steps `Yi` (shape `(Ndiffuse-1, Hsample, Nu)`):
- Sets up static plot of parking lot, obstacles, optional `env.xref` demo path, and start/goal markers.
- For every `step % frame_skip == 0`, rolls out `Yi[step]` sequentially using `step_env_jit`, updates scatter/line plots with increasing opacity, saves `frame_%04d.png`, and exports `denoising_process_animation.mp4` via `export_video`.
