# `mbd/robust/planner_wrapper.py`

**Source file:** [mbd/robust/planner_wrapper.py](../../../mbd/robust/planner_wrapper.py)

## Overview
Implements **Stage 3 (Planner Interface): `SafeMPDPlanner`**, a NumPy-facing wrapper around `run_diffusion` ([mbd/planners/mbd_planner.py](../planners/mbd_planner.md)). Enables CPU-based online execution and RMPS replanning ([mbd/robust/rmps_runner.py](rmps_runner.md)) to invoke `x_plan, u_plan = planner.plan(x_init, x_goal)` cleanly without manually managing JAX PRNG keys, static-self method JIT caches, or GUI side effects.

---

## Helpers & Class

### `_clear_method_caches(env, method_names)` & `_format_state_for_env(vec, nx)`
- Clears class-level `@partial(jax.jit, static_argnums=(0,))` caches on specified methods across `type(env).__mro__` (only needed if `x_goal` changes).
- Pads or slices state vectors cleanly to match `env.observation_size`.

### `class SafeMPDPlanner`
- **`__init__(self, config: MBDConfig | dict, env, restore_env_x0: bool = True)`**: Stores `config` (converting dicts via `dict_to_config_obj`), `env`, `restore_env_x0`, and tracking counters (`plan_count`, `last_reward`, `last_timing_info`).
- **`plan(self, x_init, x_goal=None, seed=None) -> tuple[np.ndarray, np.ndarray]`**:
  1. Updates `env.xg` only if `x_goal` differs from `env.xg` (clearing reward/step caches and `mbd_planner`'s cache).
  2. Builds a headless copy of `config` (`render=False`, `show_animation=False`, `save_animation=False`, `save_denoising_animation=False`, seed `config.seed + plan_count`).
  3. Passes `x_init` **dynamically** via `run_diffusion(args=plan_cfg, env=self.env, x_init=new_x0 if x0_changed else None)` without mutating `env.x0` or invalidating `env.reset` / rollout JIT caches — eliminating XLA recompilation on online replans.
  4. Increments `plan_count` and returns NumPy arrays `(x_plan, u_plan)` of shapes `(H + 1, Nx)` and `(H, Nu)`.
