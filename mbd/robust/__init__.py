"""
Robust Safe-MPD (RMPS) subpackage.

Exports:
- Phase 1: Offline reachable tube calibration (`tube_calibration.py`)
- Phase 2: Pure-JAX vectorized constrained iLQR backup (`ilqr_backup.py`)
- Phase 3: Clean NumPy planner wrapper (`planner_wrapper.py`) & CPU CasADi
  Tube-Based Robust NMPC Backup controller (`nmpc_backup.py`)
- Phase 4: Online closed-loop RMPS runner & diagnostics (`rmps_runner.py`)
"""

from .ilqr_backup import (
    ILQRBackupParams,
    acc_tt2d_jacobians,
    acc_tt2d_step,
    generate_braking_warmstart,
    solve_ilqr_backup,
)
from .nmpc_backup import RobustNMPCBackup
from .planner_wrapper import SafeMPDPlanner
from .rmps_runner import (
    compute_min_obstacle_clearance_np,
    plot_rmps_diagnostics,
    run_rmps_episode,
    save_rmps_results,
)
from .tube_calibration import (
    calibrate_tubes,
    compute_tvlqr_gains,
    default_tube_path,
    generate_nominal_maneuver,
    load_tubes,
    save_tubes,
)

__all__ = [
    "ILQRBackupParams",
    "RobustNMPCBackup",
    "SafeMPDPlanner",
    "acc_tt2d_jacobians",
    "acc_tt2d_step",
    "calibrate_tubes",
    "compute_min_obstacle_clearance_np",
    "compute_tvlqr_gains",
    "default_tube_path",
    "generate_braking_warmstart",
    "generate_nominal_maneuver",
    "load_tubes",
    "plot_rmps_diagnostics",
    "run_rmps_episode",
    "save_rmps_results",
    "save_tubes",
    "solve_ilqr_backup",
]
