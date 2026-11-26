"""
Robust Safe-MPD (RMPS) subpackage.

Phase 1 exports offline reachable disturbance tube calibration utilities.
"""

from .tube_calibration import (
    calibrate_tubes,
    compute_tvlqr_gains,
    default_tube_path,
    generate_nominal_maneuver,
    load_tubes,
    save_tubes,
)

__all__ = [
    "calibrate_tubes",
    "compute_tvlqr_gains",
    "default_tube_path",
    "generate_nominal_maneuver",
    "load_tubes",
    "save_tubes",
]
