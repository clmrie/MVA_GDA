# Geometric Data Analysis (GDA) Project: Vector and Heat Methods

This repository contains the complete implementation and experimental analysis for a project reproducing and critically evaluating the **Scalar Heat Method** (SHM) and the **Vector Heat Method** (VHM) for efficient geometry processing.

The core goal is to replace complex, slow path-tracing algorithms with sparse linear algebra solutions to compute geodesic distance, parallel transport, and geometric centers (Karcher Means) on curved surfaces.

* **Scalar Geodesics:** [The Heat Method for Distance Computation](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)
* **Vector Transport:** [The Vector Heat Method (arXiv)](https://arxiv.org/pdf/1805.09170)

## Key Features

* **Scalar Heat Method (SHM):** Efficiently computes geodesic distance using two linear Poisson equations.
* **Vector Heat Method (VHM):** Extends the method to compute parallel transport of tangent vectors using the complex-valued **Connection Laplacian** ($L^\nabla$).
* **Karcher Mean Computation:** Solves the geometric barycenter problem using an iterative descent algorithm enabled by the Logarithmic Map (VHM + SHM).
* **Critical Analysis:** Includes experiments stress-testing the methods against noise, parameter sensitivity, and performance bottlenecks on dynamic meshes.

## Repository Structure

The project is structured to cleanly separate the core physics engine from experimental code and documentation:

| Directory | Content | Description |
| :--- | :--- | :--- |
| `src/` | `mesh.py`, `heat_method.py`, `vector_method.py`, `operators/` | The core Python library implementing the discrete geometry and heat method physics. |
| `experiments/` | `time_breakdown.py`, `robustness_noise.py`, etc. | Scripts used for generating the critical analysis figures and performance metrics. |
| `data/` | `bunny/`, `armadillo.ply`, etc. | Input mesh files used for testing and visualization. |
| `results/` | `figure2_logmap.png`, `figure3_karcher.png`, `plots/` | All code-generated output images and quantitative plots. |
| `Report/` | `main.tex`, `figures/` | The final LaTeX source document and included figures. |
| `visualize.py` | Main entry point | The Polyscope-based interactive viewer with view-toggling UI. |


