# `render.py`

**Source file:** [render.py](../render.py)

## Overview
Standalone visualization utility script that loads saved NumPy trajectory artifacts (`nominal_trajectory_states.npy` / `trajectory_states.npy`, `perturbed_trajectory_states.npy`, `final_actions.npy`, `perturbed_actions.npy`, `mu_0ts.npy`) from a results directory and plots:
1. Both the **Nominal Trajectory** and the **Perturbed Trajectory** in $(x, y)$ with start/end markers and heading quiver arrows.
2. Each component of the nominal denoised control sequence `final_actions` and perturbed control sequence `perturbed_actions` over the planning horizon.

## Module-Level Execution (No Functions Defined)
This script runs directly at top level without function definitions:

| Lines | Behavior |
|---|---|
| 6 | Reads `results_dir` from `sys.argv[1]` (defaults to current working directory `.`). |
| 8–10 | Loads `trajectory_states.npy`, `final_actions.npy`, and `mu_0ts.npy`. |
| 12–14 | Prints array shapes to stdout. |
| 16–40 | Left subplot: plots `(x, y)` trajectory, green start / red end scatter points, and heading arrows via `ax.quiver` using `states[:, 2]` ($\theta_1$). |
| 42–52 | Right subplot: plots each control channel `actions[:, i]` (`u[0]`, `u[1]`, …) over horizon steps, applies `tight_layout()`, and calls `plt.show()`. |

## Notes for Extensions
- **New vehicle class**: Works automatically for any state dimension $\ge 3$ as long as columns `0, 1, 2` correspond to `(px, py, theta1)`, and plots all `actions.shape[1]` control channels automatically.
