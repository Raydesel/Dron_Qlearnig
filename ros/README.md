# ROS workspace (catkin) — integration notes

Your real workspace stays at **`~/catkin_ws`** on Ubuntu 20.04. This `ros/` directory is **not** a full catkin workspace.

## Bundled snapshot: `catkin_overlay/`

**[catkin_overlay/README.md](catkin_overlay/README.md)** — a copy of your **`map2gazebo`** package and a **`rotors_gazebo_overlay`** tree (worlds, `models/map`, launch, custom `src`, `collisions_detector.py`, `CMakeLists.txt`) to merge into an official RotorS checkout.

The Python code in the parent folder connects to RotorS as follows.

## What lives in `~/catkin_ws` (typical)

| Path | Role |
|------|------|
| `src/rotors_simulator/` | RotorS stack; **`rotors_gazebo`** holds worlds, models, resources |
| `rotors_gazebo/resource/waypoints/` | Output target for `Trayectorias.py` (waypoint files) |
| `rotors_gazebo/worlds/` | Custom `.world` files (e.g. farmacia, bodega) |
| `rotors_gazebo/models/` | Gazebo models / meshes for your maps |

## Syncing only your custom files into Git (optional)

If you want version control for **your** worlds and scripts without vendoring all of RotorS:

```bash
# Example: copy selected assets from your machine into a small overlay repo or branch
rsync -av --relative ~/catkin_ws/src/rotors_simulator/rotors_gazebo/worlds/./ \
  /path/to/your/backup/rotors_gazebo/worlds/
```

Then commit that smaller tree elsewhere, or attach it as a separate repository. Do not commit `build/`, `devel/`, or large bags of experimental waypoints unless you explicitly need them for a paper artifact.

## Environment variable

The Python pipeline respects **`ROTORS_WAYPOINTS_DIR`** so you are not tied to a hardcoded username. See `docs/aliases_qlearningdron.txt` and `docs/ROS_CATKIN_AND_WORKFLOW.md`.
