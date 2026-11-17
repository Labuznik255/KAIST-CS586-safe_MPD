# `tests/test_mbd.py`

**Source file:** [tests/test_mbd.py](../../tests/test_mbd.py)

## Overview
Minimal end-to-end example script demonstrating how to run Safe-MPD on a custom `"navigation"` environment with user-specified rectangular and circular obstacles, and how to configure the 4 perturbation modes (Levels 0, 1, 2, 3) and noise parameters via `MBDConfig`, `mbd.envs.get_env(...)`, or `env.set_perturbation(...)`.

## Functions

### `main()` (lines 18–154)
Executes a 6-step workflow:
1. Creates `MBDConfig` (`env_name="acc_tt2d"`, `case="navigation"`, `Nsample=20000`, `Hsample=50`, `Ndiffuse=100`) along with perturbation settings (`perturbation_level=1`, `perturbation_seed=None`, `control_noise_std=(0.05, 0.05)`, `param_uncertainty_std=0.10`, `non_gaussian_dist="student_t"`, `non_gaussian_df=2.5`, `state_disturbance_scale=0.05`).
2. Creates base `Env(width=32.0, height=32.0, case="navigation")`.
3. Instantiates `env = mbd.envs.get_env(...)` forwarding the perturbation parameters (and shows commented `env.set_perturbation(...)` override).
4. Adds 5 rectangular obstacles via `env.set_rectangle_obs(...)` and 4 circular obstacles via `env.set_circle_obs(...)`.
5. Sets initial pose `(-10.0, 5.0, 0.0, 0.0)` and goal pose `(5.0, -5.0, -pi/2, -pi/2)`.
6. Prints active perturbation info (`perturbation_level`, `nominal_params`, `perturbed_params`), runs `run_diffusion(args=config, env=env)`, and prints final nominal vs. perturbed states and $(x, y)$ deviation.
