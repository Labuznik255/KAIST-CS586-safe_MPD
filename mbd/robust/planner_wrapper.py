"""
Clean Python/NumPy Wrapper around JAX Safe-MPD (`run_diffusion`).

Allows the CPU-based online RMPS execution loop (`rmps_runner.py`) to call
`x_plan, u_plan = planner.plan(x_init, x_goal)` and receive NumPy arrays without
managing JAX PRNG keys, static method JIT caches, or GUI side effects.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from mbd.planners.mbd_planner import MBDConfig, clear_jit_cache, dict_to_config_obj, run_diffusion


def _clear_method_caches(env: Any, method_names: tuple[str, ...]) -> None:
    """Clear class-level `@partial(jax.jit, static_argnums=(0,))` caches on `env`."""
    for cls in type(env).__mro__:
        for name in method_names:
            fn = cls.__dict__.get(name)
            if fn is not None and hasattr(fn, "clear_cache"):
                fn.clear_cache()


def _format_state_for_env(vec: Any, nx: int) -> np.ndarray:
    """Coerce a state/goal vector to shape `(nx,)` (padding `[v=0, delta=0]` if 4D -> 6D)."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == nx:
        return arr
    if arr.size == 4 and nx == 6:
        return np.concatenate([arr, np.zeros(2, dtype=np.float32)], axis=0)
    if arr.size > nx:
        return arr[:nx]
    out = np.zeros(nx, dtype=np.float32)
    out[: arr.size] = arr
    return out


class SafeMPDPlanner:
    """
    Stateless-feeling Python/NumPy planner interface wrapping `run_diffusion`.

    Handles updating `env.x0` and `env.xg` across online replans while selectively
    invalidating JAX static-self JIT caches so replanning from a new `x_init`
    towards the same `x_goal` avoids unnecessary XLA recompilations.
    """

    def __init__(
        self,
        config: MBDConfig | dict[str, Any],
        env: Any,
        restore_env_x0: bool = True,
    ) -> None:
        if isinstance(config, dict):
            config = dict_to_config_obj(config)
        self.config: MBDConfig = config
        self.env = env
        self.restore_env_x0 = bool(restore_env_x0)

        self.plan_count: int = 0
        self.last_reward: float | None = None
        self.last_timing_info: dict[str, Any] | None = None

    def plan(
        self,
        x_init: np.ndarray | Any,
        x_goal: np.ndarray | Any | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Plan a nominal trajectory from `x_init` to `x_goal` using Safe-MPD.

        Args:
            x_init: Initial state vector (shape `(6,)` for `AccTractorTrailer2d`).
            x_goal: Optional goal state vector; defaults to current `self.env.xg`.
            seed: Optional RNG seed override; defaults to `config.seed + plan_count`.

        Returns:
            x_plan: Nominal planned states of shape `(H + 1, Nx)` as `np.ndarray`.
            u_plan: Nominal planned normalized controls of shape `(H, Nu)` as `np.ndarray`.
        """
        nx = int(self.env.observation_size)
        orig_x0 = np.asarray(self.env.x0, dtype=np.float32).copy()
        orig_xg = np.asarray(self.env.xg, dtype=np.float32).copy()

        new_x0 = _format_state_for_env(x_init, nx)
        new_xg = orig_xg if x_goal is None else _format_state_for_env(x_goal, nx)

        x0_changed = not np.allclose(new_x0, orig_x0, atol=1e-6)
        xg_changed = not np.allclose(new_xg, orig_xg, atol=1e-6)

        if x0_changed:
            self.env.x0 = jnp.asarray(new_x0, dtype=jnp.float32)
            # `reset` bakes `self.x0` via static_argnums=(0,); `step` takes state dynamically
            _clear_method_caches(self.env, ("reset",))

        if xg_changed:
            self.env.xg = jnp.asarray(new_xg, dtype=jnp.float32)
            _clear_method_caches(
                self.env,
                ("get_reward", "get_terminal_reward", "step", "_step_internal"),
            )
            clear_jit_cache()

        run_seed = int(self.config.seed + self.plan_count) if seed is None else int(seed)
        plan_cfg = replace(
            self.config,
            seed=run_seed,
            render=False,
            show_animation=False,
            save_animation=False,
            save_denoising_animation=False,
        )

        try:
            rew_final, Y0, trajectory_states, timing_info = run_diffusion(
                args=plan_cfg,
                env=self.env,
            )
        finally:
            if self.restore_env_x0 and x0_changed:
                self.env.x0 = jnp.asarray(orig_x0, dtype=jnp.float32)
                _clear_method_caches(self.env, ("reset",))

        self.plan_count += 1
        self.last_reward = float(rew_final)
        self.last_timing_info = timing_info

        x_plan = np.asarray(trajectory_states, dtype=np.float64)
        u_plan = np.asarray(Y0, dtype=np.float64)
        return x_plan, u_plan
