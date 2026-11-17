# `mbd/robots/__init__.py`

**Source file:** [mbd/robots/__init__.py](../../../mbd/robots/__init__.py)

## Overview
Subpackage initializer for `mbd.robots`. Re-exports all vehicle dynamics/environment classes.

## Exports
- `TractorTrailer2d` from [tt2d.py](../../../mbd/robots/tt2d.py)
- `AccTractorTrailer2d` from [acc_tt2d.py](../../../mbd/robots/acc_tt2d.py)
- `KinematicBicycle2d` from [kinematic_bicycle2d.py](../../../mbd/robots/kinematic_bicycle2d.py)
- `NTrailer2d` from [n_trailer2d.py](../../../mbd/robots/n_trailer2d.py)

## Where to Add a New Vehicle Class
After creating your new vehicle module (e.g. `mbd/robots/my_vehicle.py`), import its class here and add its name to `__all__` (lines 1–11). Then register it in [mbd/envs/__init__.py](../../../mbd/envs/__init__.py).
