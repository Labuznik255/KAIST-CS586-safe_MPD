# `mbd/envs/env.py`

**Source file:** [mbd/envs/env.py](../../../mbd/envs/env.py)

## Overview
Defines the `Env` class, which manages the 2D workspace geometry (dimensions, plot limits), parking-lot layouts, and circular/rectangular obstacle storage for `"parking"` and `"navigation"` scenarios. Robot classes hold an `Env` instance as `self.env`.

---

## Class: `Env` (lines 14–299)

### `__init__(self, width=32.0, height=32.0, case="parking", parking_config=None, resolution=0.1)` (lines 15–43)
Initializes the 2D workspace bounds `x_range = (-width/2, width/2)` and `y_range = (-height/2, height/2)`.
- If `case == "parking"`: uses `parking_config` or `get_default_parking_config()`, then populates `self.obs_circles` and `self.obs_rectangles` via `set_obs_circle_parking` and `set_obs_rectangle_parking`.
- If `case == "navigation"`: initializes empty obstacle lists (populated later via `set_rectangle_obs` / `set_circle_obs`).
- Raises `ValueError` for unknown `case`.
- If `obs_circles` is empty, inserts a dummy zero-radius circle `[[0.0, 0.0, 0.0]]` so JAX obstacle arrays remain non-empty for `vmap` / shape consistency.

### `get_default_parking_config(self)` (lines 44–57)
Returns a dictionary specifying a 2-row × 8-column parking lot (`space_width=3.5`, `space_length=7.0`, `parking_y_offset=4.0`, `occupied_spaces=[1, 2, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15]`, `target_spaces=[3, 11]`, `obstacle_radius=1.0`).

### `set_obs_boundary(width, height)` (`@staticmethod`, lines 59–64)
Placeholder returning an empty list `[]` (no boundary walls by default).

### `set_obs_rectangle()` (`@staticmethod`, lines 66–69)
Deprecated static placeholder returning `[]`.

### `set_obs_rectangle_parking(self)` (lines 71–79)
Returns default top and bottom horizontal boundary wall rectangles for the parking scenario: `[0.0, 15.0, 30.0, 1.0, 0.0]` and `[0.0, -14.0, 30.0, 1.0, 0.0]` (format `[x_center, y_center, width, height, rotation]`).

### `set_obs_circle_parking(self, parking_config)` (lines 81–119)
Converts 1-indexed `occupied_spaces` in `parking_config` into circular obstacles. For each occupied space, places three circles of `obstacle_radius` at `(space_center_x, space_center_y)` and `±2.0` m vertically.

### `get_parking_space_center(self, space_num)` (lines 121–148)
Computes `(space_center_x, space_center_y)` for a 1-indexed parking space number. Raises `ValueError` if `self.case != "parking"`.

### `get_default_init_pos(self, case)` (lines 150–160)
Returns default 4D initial state `jnp.array([start_x, start_y, theta1, theta2])` near the top-left corner (`x_range[0] + 2.0`, `y_range[1] - 2.0`, `theta1=0.1`, `theta2=0.1`) for `"parking"`. Raises `ValueError` otherwise.

### `get_default_goal_pos(self, case)` (lines 162–172)
Returns default 4D goal state `[goal_x, goal_y, -np.pi/2, -np.pi/2]` centered at `parking_config['target_spaces'][0]` for `"parking"`. Raises `ValueError` otherwise.

### `get_obstacles(self)` (lines 174–180)
Returns `{'circles': jnp.ndarray (N, 3), 'rectangles': jnp.ndarray (M, 5)}`, using empty `(0, 3)` / `(0, 5)` arrays when empty.

### `get_circular_obstacles(self)` (lines 182–184)
Returns `jnp.ndarray` of circular obstacles `(N, 3)`.

### `get_rectangular_obstacles(self)` (lines 186–188)
Returns `jnp.ndarray` of rectangular obstacles `(M, 5)`.

### `is_in_bounds(self, x, y)` (lines 190–193)
Returns Python `bool` checking whether `(x, y)` lies inside `self.x_range` and `self.y_range`.

### `get_plot_limits(self)` (lines 195–197)
Returns `(self.x_range, self.y_range)`.

### `print_parking_layout(self)` (lines 199–227)
Logs an ASCII grid of the parking lot (`O` = Occupied, `E` = Empty, `T` = Target) at `DEBUG` level.

### `set_rectangle_obs(self, obstacles, coordinate_mode="left-top", padding=0.0)` (lines 229–266)
Parses 4-element `[x, y, width, height]` or 5-element `[x, y, width, height, rotation]` obstacle lists, converts `"left-top"` coordinates to center coordinates if needed (`center_x = x + width/2`, `center_y = y - height/2`), adds `2 * padding` to width and height, stores as `self.obs_rectangles` (NumPy array), and returns it.

### `set_circle_obs(self, obstacles, padding=0.0)` (lines 268–292)
Parses `[x, y, radius]` lists, adds `padding` to `radius`, stores as `self.obs_circles` (NumPy array of shape `(N, 3)`), and returns it.

### `set_plot_limits(self, x_range, y_range)` (lines 294–298)
Updates `self.x_range` and `self.y_range` for plotting.

---

## Extension Notes
- **New vehicle class**: `Env` is vehicle-agnostic except that `get_default_init_pos` / `get_default_goal_pos` return 4D arrays `[px, py, theta1, theta2]`. Subclasses of `TractorTrailer2d` slice or extend these 4D arrays in their `__init__` (see `KinematicBicycle2d` and `AccTractorTrailer2d`).
- **Perturbations**: Environmental wind fields or surface friction zones (spatially varying perturbations) could be stored as attributes on `Env` if perturbations depend on `(x, y)` world position.
