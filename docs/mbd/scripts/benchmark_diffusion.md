# `mbd/scripts/benchmark_diffusion.py`

**Source file:** [mbd/scripts/benchmark_diffusion.py](../../../mbd/scripts/benchmark_diffusion.py)

## Overview
Timing benchmark script designed to measure **pure diffusion runtime** (excluding JAX JIT compilation, warmup, demo generation, and rendering overhead) across multiple configurations and repeated trials.

## Functions

### `benchmark_diffusion(args=None, env=None, num_trials=5, warmup_trials=1)` (lines 21–132)
Runs `warmup_trials` discarded runs of `run_diffusion(args=args, env=env)` followed by `num_trials` timed runs.
- **Returns**: Dictionary containing `num_trials`, summary statistics (`mean`, `std`, `min`, `max`, `median`) for `pure_diffusion_time`, `total_time`, `overhead_time`, `compilation_time`, and `rewards`, plus raw `all_timing_info` and `all_rewards`.
- Prints a formatted benchmark summary table.

### `benchmark_parking_scenario()` (lines 135–253)
Benchmarks `"tt2d"` in `"parking"` mode (`motion_preference=-1`, `enable_demo=True`, `render=False`) across three `(Nsample, Hsample, Ndiffuse)` sizes:
1. `"Small (Fast)"`: `(5000, 30, 100)`
2. `"Medium"`: `(10000, 40, 120)`
3. `"Large (Original)"`: `(20000, 50, 150)`
- Constructs `env` via `mbd.envs.get_env(...)`, calls `env.set_init_pos(dx=-3.0, dy=1.0, theta1=0, theta2=0)` and `env.set_goal_pos(theta1=jnp.pi/2, theta2=jnp.pi/2)`, runs `benchmark_diffusion(..., num_trials=3, warmup_trials=1)`, and prints a comparison table.
- **Returns**: `results` dict keyed by configuration name.

### `__main__` block (lines 256–269)
Runs `benchmark_parking_scenario()` and prints summary notes.

## Extension Notes
- To benchmark a new vehicle class or perturbed tractor-trailer model, update `base_config.env_name` (line 143) and forward any new parameters in `mbd.envs.get_env(...)` (lines 190–219). Note that `clear_jit_cache()` is not called between `test_configs`, because `Nsample`/`Hsample`/`Ndiffuse` are handled via `reverse_once` args/recompilation—if `env` class changes between configs, call `clear_jit_cache()`.
