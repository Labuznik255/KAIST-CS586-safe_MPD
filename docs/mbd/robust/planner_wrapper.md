# `mbd/robust/planner_wrapper.py`

**Source file:** [mbd/robust/planner_wrapper.py](../../../mbd/robust/planner_wrapper.py)

## Overview
Implements **Stage 3 (Planner Interface): `SafeMPDPlanner`**, a NumPy-facing wrapper around `run_diffusion` (`mbd/planners/mbd_planner.py`). Enables CPU-based online execution and RMPS replanning (`rmps_runner.py`) to invoke `x_plan, u_plan = planner.plan(x_init, x_goal)` cleanly without manually managing JAX PRNG keys, static-self method JIT caches, or GUI side effects.

---

## Helpers & Class

### `_clear_method_caches(env, method_names)` & `_format_state_for_env(vec, nx)` (lines 20–40)
- Clears class-level `@partial(jax.jit, static_argnums=(0,))` caches on specified methods across `type(env).__mro__`.
- Pads or slices state vectors cleanly to match `env.observation_size`.

### `class SafeMPDPlanner` (lines 43–135)
- **`__init__(self, config: MBDConfig | dict, env, restore_env_x0: bool = True)`** (lines 52–66): Stores `config` (converting dicts via `dict_to_config_obj`), `env`, `restore_env_x0`, and tracking counters (`plan_count`, `last_reward`, `last_timing_info`).
- **`plan(self, x_init, x_goal=None, seed=None) -> tuple[np.ndarray, np.ndarray]`** (lines 68–135):
  1. Updates `env.x0` (clearing only `reset`'s JIT cache so main rollout kernels stay compiled across replans to the same `xg`) and updates `env.xg` if changed (clearing reward/step caches and `mbd_planner`'s cache).
  2. Builds a headless copy of `config` (`render=False`, `show_animation=False`, `save_animation=False`, `save_denoising_animation=False`, seed `config.seed + plan_count`).
  3. Calls `run_diffusion(args=plan_cfg, env=self.env)`, restores `env.x0` if `restore_env_x0=True`, increments `plan_count`, and returns NumPy arrays `(x_plan, u_plan)` of shapes `(H + 1, Nx)` and `(H, Nu)`.
