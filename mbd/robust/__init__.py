"""
Robust Safe-MPD (RMPS) subpackage.

Exports offline reachable tube calibration utilities (Phase 1) and the pure-JAX
vectorized iLQR backup solver (Phase 2).
"""

from .ilqr_backup import (
    ILQRBackupParams,
    acc_tt2d_jacobians,
    acc_tt2d_step,
    generate_braking_warmstart,
    solve_ilqr_backup,
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
    "acc_tt2d_jacobians",
    "acc_tt2d_step",
    "calibrate_tubes",
    "compute_tvlqr_gains",
    "default_tube_path",
    "generate_braking_warmstart",
    "load_tubes",
    "save_tubes",
    "solve_ilqr_backup",
]
