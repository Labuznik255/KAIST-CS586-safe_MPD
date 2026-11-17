# `tests/test_mbd.py`

**Source file:** [tests/test_mbd.py](../../tests/test_mbd.py)

## Overview
Minimal end-to-end example script demonstrating how to run Safe-MPD on a custom `"navigation"` environment with user-specified rectangular and circular obstacles, and how to configure the 4 perturbation modes (Levels 0, 1, 2, 3) and noise parameters via `MBDConfig`, `mbd.envs.get_env(...)`, or `env.set_perturbation(...)`.

## Functions

### `main()` (lines 18–118)
Executes a 6-step workflow:
1. Creates `MBDConfig(env_name="acc_tt2d", case="navigation", Nsample=20000, Hsample=50, Ndiffuse=100, render=True, show_animation=True, ...)`.
2. Creates base `Env(width=32.0, height=32.0, case="navigation")`.
3. Instantiates the robot environment via `mbd.envs.get_env(config.env_name, case=config.case, env_config=env_config, dt=config.dt, H=config.Hsample)`.
4. Adds 5 rectangular obstacles via `env.set_rectangle_obs(..., coordinate_mode="center", padding=0.0)` and 4 circular obstacles via `env.set_circle_obs(..., padding=0.0)`.
5. Sets initial pose `env.set_init_pos(x=-10.0, y=5.0, theta1=0.0, theta2=0.0)` and goal pose `env.set_goal_pos(x=parking_x, y=parking_y, theta1=-jnp.pi/2, theta2=-jnp.pi/2)`.
6. Calls `run_diffusion(args=config, env=env)` and prints final reward and timings.

## Extension Notes
- **Testing a new vehicle class**: Change `env_name` in `MBDConfig` (line 23) and pass any new vehicle parameters to `mbd.envs.get_env(...)` (lines 43–49).
- **Testing perturbed tractor-trailer models**: Forward perturbation parameters in `mbd.envs.get_env(...)` or test closed-loop/perturbed rollout on the returned `Y0`.
