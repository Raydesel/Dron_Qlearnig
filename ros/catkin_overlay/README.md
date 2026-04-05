# Catkin overlay (copied from your `~/catkin_ws`)

This folder snapshots **ROS packages and `rotors_gazebo` customizations** from your machine so the thesis pipeline is reproducible without committing all of `~/catkin_ws`.

## Contents

| Path | Description |
|------|-------------|
| `map2gazebo/` | Full **map2gazebo** catkin package (map → Gazebo workflow; MIT license per upstream `package.xml`). |
| `rotors_gazebo_overlay/` | **Partial** tree meant to be **merged on top of** the official [ethz-asl/rotors_simulator](https://github.com/ethz-asl/rotors_simulator) `rotors_gazebo` package. Includes custom **worlds**, **`models/map`** (DAE meshes), **launch** files, **`src/`** C++ nodes (`waypoint_publisher_file`, `reset_world`, `test_map_wp`, etc.), **`scripts/collisions_detector.py`**, and your **`CMakeLists.txt`** / **`package.xml`**. |
| `rotors_gazebo_overlay/resource/waypoints/` | Empty except `.gitkeep`; Python writes waypoint files here after you merge into the workspace. |

**Not copied** (too large or generated): stock RotorS assets under `models/` other than `map/`, `build/`, `devel/`, and previous experiment outputs under `resource/waypoints/`.

## Prerequisites

- Ubuntu 20.04, **ROS Noetic**, Gazebo compatible with RotorS.
- Full **rotors_simulator** stack cloned under `~/catkin_ws/src/` (same layout as ETHZ).

## One-time setup: merge overlay into `rotors_gazebo`

From this repository root, with your workspace **not** sourced (optional):

```bash
ROTOR_PKG="$HOME/catkin_ws/src/rotors_simulator/rotors_gazebo"
OVERLAY="$(pwd)/ros/catkin_overlay/rotors_gazebo_overlay"

# Backup (recommended)
cp -a "$ROTOR_PKG/CMakeLists.txt" "$ROTOR_PKG/CMakeLists.txt.bak"

# Merge: overlay files win on name clashes
rsync -a "$OVERLAY/" "$ROTOR_PKG/"
```

Then add **map2gazebo** if you do not already have it:

```bash
rsync -a "$(pwd)/ros/catkin_overlay/map2gazebo/" \
  "$HOME/catkin_ws/src/map2gazebo/"
```

Build:

```bash
cd ~/catkin_ws
catkin build rotors_gazebo map2gazebo   # or catkin_make
source devel/setup.bash
```

## If `CMakeLists.txt` conflicts after a RotorS upgrade

Your saved `CMakeLists.txt` defines extra executables (`waypoint_publisher_file`, `reset_world`, …). If upstream ETHZ changes the same file, **merge manually** instead of blind overwrite: keep upstream structure and re-apply your `add_executable` / `target_link_libraries` blocks.

## Waypoints and Python

After merging, set (or keep default):

```bash
export ROTORS_WAYPOINTS_DIR="$HOME/catkin_ws/src/rotors_simulator/rotors_gazebo/resource/waypoints"
```

Run `Trayectorias.py` from the parent repo so generated paths match the IRIS waypoint publisher.
