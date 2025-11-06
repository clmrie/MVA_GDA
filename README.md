# MVA Geometric Analysis

This repository contains reference implementations of the Heat Method and the Vector Heat Method for computing geodesic distances on triangle meshes, along with a few helper utilities for loading meshes and assembling standard discrete differential geometry operators.

## Prerequisites

Create a Python environment (3.9 or later is recommended) and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts rely on the Stanford bunny model that is already included under `data/bunny/reconstruction/bun_zipper.ply`.

## Running the sanity checks

A lightweight regression script is provided under `tests/run_checks.py`. It loads the bunny mesh, assembles the discrete operators, and evaluates both the scalar heat method and the vector heat method implementations. The script prints basic diagnostics (symmetry, null-space checks) and reports the runtime of each solver along with the agreement between the resulting distance fields.

Run it from the repository root:

```bash
python -m tests.run_checks
```

The script terminates successfully if no assertion errors are raised. Typical output looks like this:

```
Mesh: (34834, 3) (69662, 3)
L symmetric: True
||L·1||: 2.8e-08
M diag min/max: 7.4e-06 2.5e-04
max ||∑_a ∇φ_a|| per face: 1.2e-10
HeatMethod OK: t=2.13e-03, phi range=(0,0.312), runtime=0.47s
VectorHeatMethod OK: t=2.13e-03, phi range=(0,0.313), runtime=0.94s, rel err vs heat=3.6e-03
```

These values will vary slightly depending on the SciPy solver version, but the diagnostics should remain within the same magnitude.

## Additional utilities

* `generate_meshes.py` – simple helpers to resample meshes.
* `experiments/` – exploratory notebooks and scripts used during development.

Feel free to adapt the workflow to your own meshes; both the heat and vector heat entry points accept any watertight triangle mesh in OBJ or PLY format.
