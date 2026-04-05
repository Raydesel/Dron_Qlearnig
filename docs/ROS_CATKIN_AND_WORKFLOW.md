# ROS, catkin workspace, and this repository

## Should `catkin_ws` live inside this Git repo?

**Usually no.** A typical `catkin_ws` contains:

- `build/`, `devel/`, `logs/` — thousands of generated files (do not commit)
- Upstream packages such as **RotorS** (`rotors_simulator`) — large, already versioned upstream
- **Runtime outputs** under `rotors_gazebo/resource/waypoints/` — experiment artifacts that bloat the repo

**Recommended layout on your PC (what you already have):**

```text
~/Documents/Python2ROS/Dron_Qlearnig/   ← this repository (Python + docs)
~/catkin_ws/                              ← ROS workspace (separate)
    src/
        rotors_simulator/                 ← RotorS + your worlds/models/scripts
        ...
```

This repository documents how to connect the two; it does not need to duplicate the whole workspace.

## Ubuntu 20.04 and ROS Noetic

1. Install [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) and the usual build tools (`catkin_tools` or `catkin_make`).
2. Place **RotorS** under `~/catkin_ws/src/` (clone from [ETHZ ASL rotors_simulator](https://github.com/ethz-asl/rotors_simulator) or your fork), then merge your **custom** `rotors_gazebo` content (worlds, meshes, launch files, scripts) as you did for the thesis.
3. Build and source:

```bash
cd ~/catkin_ws
catkin build   # or: catkin_make
source devel/setup.bash
```

Add to `~/.bashrc`:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

## Linking Python output to RotorS waypoints

`Trayectorias.py` writes under the RotorS **waypoints** resource directory so ROS nodes can load the generated `.txt` / `.csv` files.

- **Default:** `~/catkin_ws/src/rotors_simulator/rotors_gazebo/resource/waypoints/`
- **Override:** set environment variable `ROTORS_WAYPOINTS_DIR` to any folder (with or without trailing slash); the script normalizes it.

Example:

```bash
export ROTORS_WAYPOINTS_DIR="$HOME/catkin_ws/src/rotors_simulator/rotors_gazebo/resource/waypoints"
python3 Trayectorias.py
```

See also `docs/aliases_qlearningdron.txt`.

## Snapshot included in this repository

The folder **`ros/catkin_overlay/`** in the parent Git repo contains:

- the **`map2gazebo`** package, and  
- a **`rotors_gazebo_overlay`** directory to **`rsync`** onto `~/catkin_ws/src/rotors_simulator/rotors_gazebo/` after cloning upstream RotorS.

See **[../ros/catkin_overlay/README.md](../ros/catkin_overlay/README.md)** for exact commands. This avoids checking in `build/`, `devel/`, or multi-megabyte stock models you did not change.

## Other ways to track ROS in Git

1. **Separate Git repo** for your ROS overlay (clone into `~/catkin_ws/src/`).  
2. **Git submodule** pointing at that overlay repo.  
3. **Manual rsync** from `ros/catkin_overlay/` whenever you refresh the snapshot.

`.gitignore` in this repository ignores typical catkin build products if you ever nest a workspace by mistake.

## Simulation reference (book)

RotorS usage and Gazebo practice were aligned with **Chapter 8** of:

Joseph, L., & Cacace, J. (2021). *Mastering ROS for Robotics Programming: Best practices and troubleshooting solutions when working with ROS*. Packt Publishing.

Keep the PDF on your machine; it is not redistributed here.

## PDF tutorials in `docs/`

Use the PDFs in this folder as your local runbooks (map → world → RotorS → Python, UAV pose, resets, methodology). They complement the top-level [README.md](../README.md).
