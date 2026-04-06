# Dron Q-Learning — Autonomous UAV navigation with reinforcement learning

This repository contains the computational pipeline developed for the master’s thesis **“Navegación autónoma de un vehículo aéreo no tripulado usando aprendizaje por refuerzo”** (Universidad Autónoma de Chihuahua, 2023), by **Raydesel Ariel Sánchez Montes**, directed by Dr. Alain Manzo Martínez.

The code builds **2D indoor trajectories** from **binary occupancy maps** using **Q-learning**, smooths them with **Bézier** or **B-spline** interpolation, and exports waypoint-style data (position, delay, yaw) for downstream use—aligned with the **ROS–Gazebo** simulation and **RotorS (IRIS)** evaluation described in the thesis.

## What this work does (thesis summary)

- **Environment**: Five indoor-style scenarios are represented as **binary maps** (PNG). The space is **discretized** into states for tabular Q-learning.
- **Planning**: **Q-learning** produces a sequence of grid cells (waypoints) from start to goal.
- **Exploration–exploitation**: Besides a **conventional** random action choice over valid moves, the thesis proposes and evaluates:
  - **Adapted Thompson sampling (MTA)** for action selection during training.
  - **Gaussian (fuzzy) weighting (DG)** over rewards for stochastic next-state choice.
  In the reported experiments, these variants achieved roughly **81% better efficiency** (time and iteration count) than the conventional approach.
- **Smoothing**: **Bézier** curves and **B-splines** (including De Boor–style evaluation in the implementation) densify the polyline into a smooth path.
- **Metrics** (thesis): Trajectory tracking in simulation used **RMSE** and **percentage error** on **XY** and **yaw**, comparing expected path to quadrotor odometry.

**Keywords (from the thesis):** autonomous navigation, indoor environments, UAVs, reinforcement learning, Q-learning.

## Data engineering angle (telemetry & reproducibility)

This project is **reinforcement learning** research, but it is also a **structured pipeline** from raw spatial inputs to evaluated outputs—similar in spirit to **sensor / telemetry analytics** and **offline experimentation**:

- **Inputs**: rasterized environment data (binary maps), discretization and graph structure, consistent naming for derived artifacts.
- **Transformations**: training runs that emit **versionable artifacts** (e.g. `Q_table.npy`, convergence CSVs), trajectory generation, spline-based densification, and timed waypoint sequences for downstream consumers.
- **Outputs**: **CSV/TXT** waypoint files, aggregated experiment tables (`Trayectorias_datos_*.csv`), and **plots** for comparison—supporting **repeatable** hyperparameter sweeps (including multiprocessing over combinations).
- **Quality / metrics**: **RMSE** and **percentage error** on position and yaw against references, aligned with how data teams quantify pipeline or model drift.

If you are hiring for **data engineering**, treat this repo as evidence of **Python data/code workflow**, **batch experimentation**, and **clear handoffs** to other systems (here: ROS/Gazebo), alongside the lakehouse and ETL work listed on my CV.

## Documentation (ROS, catkin, tutorials)

- **[docs/README.md](docs/README.md)** — index of the PDF notes (map → Gazebo → RotorS → Python, methodology, UAV pose, reset).
- **[docs/ROS_CATKIN_AND_WORKFLOW.md](docs/ROS_CATKIN_AND_WORKFLOW.md)** — how **`~/catkin_ws`** fits with this repo, Noetic on Ubuntu 20.04, and why the full workspace is usually **not** committed here.
- **[docs/aliases_qlearningdron.txt](docs/aliases_qlearningdron.txt)** — bash aliases and **`ROTORS_WAYPOINTS_DIR`** for waypoint output.
- **[ros/README.md](ros/README.md)** — integration notes for RotorS paths.
- **[ros/catkin_overlay/README.md](ros/catkin_overlay/README.md)** — **`map2gazebo`** + **`rotors_gazebo_overlay`** copied from your `~/catkin_ws` (merge instructions).

**RotorS / simulation book:** Lentin Joseph & Jonathan Cacace, *Mastering ROS for Robotics Programming* (Packt, 2021) — **Chapter 8** (as used on your setup); not bundled in the repo.

## Repository layout

| File | Role |
|------|------|
| `building_the_environment.py` | Loads a binary map (OpenCV), applies wall padding, builds **state ↔ cell** mappings, adjacency / **reward matrix**, optional **diagonal** moves and **window-based** reward shaping (`densidad`). |
| `Aprendizaje_Q.py` | Q-learning **training** (`training`, `training_thompson`, `training_gauss`), **inference** (`inference`), and high-level **`route()`** with convergence loop and optional `Q_table.npy` load. |
| `Trayectorias.py` | End-to-end experiments: builds the environment, runs `route()`, applies **B-spline** or **Bézier**, computes **RMSE / percent error** vs. degree-1 spline reference, saves plots and CSV/TXT, optional **multiprocessing** over hyperparameter combinations. |
| `bspline_with_order.py` | B-spline path generation with configurable **order** and number of points. |
| `bezier.py` | Bézier interpolation along the Q-learning waypoint chain. |
| `delay_z_yaw.py` | Builds timed sequences with **z**, **delay**, and **yaw** from interpolated XY (for waypoint publication). |
| `RMSE_bs1_bs10.py` | **RMSE** and **percentage error** between a reference (B-spline degree 1) and a higher-degree smoothed trajectory. |

## Dependencies

- Python 3
- `numpy`
- `opencv-python` (`cv2`)
- `scikit-learn` (`sklearn`) — graph from grid, metrics
- `scikit-fuzzy` (`skfuzzy`) — Gaussian membership in `training_gauss`
- `pandas`, `matplotlib`

Install example:

```bash
pip install numpy opencv-python scikit-learn scikit-fuzzy pandas matplotlib
```

## Data and configuration

1. **Binary maps**  
   Provide PNG maps consistent with the code (e.g. `Farmacia.png`, or other scenarios mentioned in the thesis such as casa, supermercado, oficina, farmacia, bodega). Pixels are thresholded to **0/1** occupancy.

2. **Gazebo-prefixed maps for plotting**  
   `guardar()` in `Trayectorias.py` may look for a `'(gz)'` + filename variant for obstacle overlay on figures—keep naming consistent with your local files if you use that path.

3. **Output directory**  
   In `process_combination()` inside `Trayectorias.py`, waypoint output goes under **`~/catkin_ws/src/rotors_simulator/rotors_gazebo/resource/waypoints/`** by default. Override with the environment variable **`ROTORS_WAYPOINTS_DIR`** (absolute path to the `waypoints` folder). See `docs/aliases_qlearningdron.txt`.

4. **Training mode** (`entrenamiento` in the combination tuple)  
   - `0` — conventional Q-learning (random valid action)  
   - `1` — Thompson-style sampling (`training_thompson`)  
   - `2` — Gaussian / fuzzy reward weighting (`training_gauss`)

5. **Other knobs**  
   `resolution` (e.g. 20, 40, 100), `diagonals`, `densidad` (window size for reward shaping), `gamma`, `alpha`, `iterations`, B-spline `order` and `points`, and start/end **state ids** must match the discretization of your map.

## Running

From the repository root (with your map PNG and paths configured):

```bash
python3 Trayectorias.py
```

The `if __name__ == '__main__'` block defines **combinations** of routes and hyperparameters and runs them with `multiprocessing.Pool`. Adjust `imagen`, `resolution`, `path`, and `combinations` to match your experiment.

Saved artifacts typically include per-run folders with **waypoint `.txt` / `.csv`**, **Q-table `.npy`**, **Q convergence CSV**, plots (`.jpg`), and an aggregated `Trayectorias_datos_*.csv`.

## Relation to ROS / Gazebo

The **thesis** validates tracking in **ROS–Gazebo** using the **IRIS** quadrotor from **RotorS**. The root Python scripts generate trajectories and waypoint files; **`ros/catkin_overlay/`** holds a portable copy of your **`map2gazebo`** package and **`rotors_gazebo`** customizations (worlds, map meshes, launch files, C++ waypoint nodes, `collisions_detector.py`) to merge into a standard [rotors_simulator](https://github.com/ethz-asl/rotors_simulator) tree—see **[ros/catkin_overlay/README.md](ros/catkin_overlay/README.md)**.

## Reference

Sánchez Montes, R. A. (2023). *Navegación autónoma de un vehículo aéreo no tripulado usando aprendizaje por refuerzo* [Master’s thesis, Universidad Autónoma de Chihuahua].

If you use this code in research, please cite the thesis and acknowledge CONACYT support as in the original document.
