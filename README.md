# Chief / Deputy RPO: 3D Simulation

A Python simulation and real-time 3D visualization of a spacecraft **rendezvous and proximity operations (RPO)** mission based in the Capstone Project from the course Spacecraft Formation
Relative Orbits offered by Colorado Boulder through Coursera. A deputy spacecraft transfers to the chief's orbit, enters a safety spiral, and then approaches a docking point fixed to the chief's body while the chief tumbles.

<img width="800" height="450" alt="Gif_video_Formation_flying" src="https://github.com/user-attachments/assets/6bd3ff8e-d865-4d19-b0e8-b964f70ec5f5" />


The whole mission is propagated up front, then replayed in a [Panda3D](https://www.panda3d.org/) window with free camera control, live plots and playback at up to 1000x speed.



## Mission overview

| Phase | Description |
|-------|-------------|
| 0 | Hohmann transfer and drift (inertial two-body propagation, two burns) |
| 1 | Safety spiral: CWH controller tracks an analytic relative-motion reference |
| 2 | Hold 25 m along the chief's body +x axis |
| 3 | Approach to 5 m |
| 4 | Approach to 2 m |

The chief's attitude is propagated with Modified Rodrigues Parameters (MRPs) and torque-free rigid-body dynamics (`Attitude_Dynamics.py`). From stage 2 onward the reference point is fixed to the chief body, so the deputy must track a point that rotates with the tumbling chief. Quaternions appear only in the visualisation layer (rendering and interpolation).

## Requirements

- Python 3.9+
- `numpy`
- `panda3d`

```bash
pip install numpy panda3d
```

## Files

| File | Purpose |
|------|---------|
| `Simulation3D.py` | Mission propagation, derived data and the Panda3D viewer |
| `Attitude_Dynamics.py` | MRP kinematics, DCM conversion, Euler's rotational equations |
| `Orbital_Mechanics.py` | Orbital elements, two-body EOM, LVLH transforms, CWH dynamics and control |
| `chief.bam` / `deputy.bam` | Spacecraft models (`.egg` also works) |
| `earth_texture.jpg` | Earth texture |

## Usage

```bash
python Simulation3D.py
```

Options:

| Flag | Meaning |
|------|---------|
| `--start SECONDS` | Start at a given simulation time |
| `--speed N` | Initial speed multiplier (1 to 1000) |
| `--size WxH` | Window size (default 1600x900) |
| `--no-cache` | Recompute instead of loading `sim_cache.npz` |
| `--paused` | Start paused |
| `--check` | Print a velocity-continuity check at burn 2 and the CWH hand-over |

The first run propagates the mission (about 10 s) and caches the result in `sim_cache.npz`. The cache rebuilds automatically when the code or `CFG` changes.

## Controls

| Key | Action |
|-----|--------|
| Space | Play / pause |
| Up / Down | Speed up / down |
| Left / Right | Seek -/+ 60 s (Shift = 10x) |
| Home | Restart |
| N / B | Next / previous stage |
| C / V / X | Cycle main camera / cycle inset camera / swap them |
| Mouse wheel, left-drag | Zoom / orbit the view under the cursor |
| F | Toggle auto-fit zoom |
| P | Plots <-> inset camera |
| A / T / L / G | Body axes / trails / labels / LVLH guides |
| H | Help overlay |
| Esc | Quit |

## Configuration

Mission parameters (orbit radii, inclination, inertia tensor, initial body rates, time step) live in the `CFG` dictionary at the top of `Simulation3D.py`.

## How it works

1. `run_simulation()` integrates the full mission with fixed-step RK4 and stores the history.
2. `SimData` derives LVLH and body-frame quantities, the deputy attitude and render quaternions.
3. Each frame, the viewer interpolates the stored state (Hermite for position, normalised lerp for attitude) and renders it using a floating origin in float64, which keeps the scene stable from Earth scale down to metres.

## Known limitations

- The stage 3 and 4 approach ramps use absolute simulation time rather than time since the stage began.
- The deputy's attitude is not simulated; it is constructed geometrically so that its +x axis points at the chief.

## Acknowledgements

Parts of this code, mainly the Panda3D visualisation layer, were developed with the assistance of an AI model (Claude). The mission design and control logic come from the author's own coursework simulation.

## License
MIT License
