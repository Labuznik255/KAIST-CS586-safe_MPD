# `mbd/planners/nmpc_planner.py`

**Source file:** [mbd/planners/nmpc_planner.py](../../../mbd/planners/nmpc_planner.py)

## Overview
Implements a standalone **Nonlinear Model Predictive Control (NMPC)** baseline planner using CasADi (`casadi.Opti()`) and IPOPT on CPU, sharing the exact same `Env`, vehicle classes (`TractorTrailer2d`, `AccTractorTrailer2d`, `KinematicBicycle2d`, `NTrailer2d`), perturbation framework (Levels 0–3), and `results/` export format as [mbd/planners/mbd_planner.py](mbd_planner.md).

Supports both:
- **Full-horizon direct multiple-shooting optimal control** (`mpc_mode=False`, default), and
- **Receding-horizon closed-loop MPC** (`mpc_mode=True`, re-solving over `mpc_horizon` at every timestep).

---

## Configuration & Key Components

### `@dataclass class NMPCConfig(MBDConfig)` (lines 39–68)
Extends `MBDConfig` for drop-in compatibility with additional IPOPT/NMPC fields:
- Solver controls: `mpc_mode` (`False`), `mpc_horizon` (`15`), `ipopt_max_iter` (`400`), `ipopt_mpc_max_iter` (`45`), `ipopt_tol` (`1e-4`), `ipopt_print_level` (`0`).
- Objective weights: `w_stage_pos`, `w_stage_heading`, `w_stage_hitch`, `w_terminal_pos`, `w_terminal_heading`, `w_terminal_vel`, `w_control`, `w_control_rate`, `w_obs_slack`.
- Safety buffers: `obstacle_safety_margin` (`0.25 m`), `hitch_safety_margin_deg` (`1.5 deg`).

### `run_nmpc(args: NMPCConfig, env)`
1. Computes a collision-free geometric warm-start path (A* grid search when obstacles block the straight line).
2. Solves either the full-horizon NLP or a receding-horizon MPC loop with exact CasADi dynamics matching `env`, smooth body-disc vs. obstacle slack constraints, and hitch limits.
3. Evaluates both nominal and perturbed (`env.step_perturbed`) rollouts in JAX, checks exact collision/hitch violations (`_check_jax_trajectory_violations`), and saves standard `results/{env_name}/` artifacts (`nominal_trajectory_states.npy`, `perturbed_trajectory_states.npy`, `final_actions.npy`, `perturbed_actions.npy`, `env_metadata.json`).
