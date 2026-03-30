# 6-DOF Kinematic State Estimation: Fully Coupled 3D Kalman Filter

This repository provides a robust C++ implementation of a fully coupled 6-dimensional Kalman Filter, explicitly designed for the state estimation of marine vehicles (e.g., Autonomous Underwater Vehicles). It estimates the 3D position ($X, Y, Z$) and dynamically extracts the hidden 3D velocity states (surge, sway, and heave rates) from noisy positional data using a Constant Velocity (CV) kinematic model.



## Methodological Overview
Unlike decoupled approximations that assume orthogonal independence, this implementation utilizes a rigorous full-matrix state-space formulation. The state vector is defined as $x = [p^T, v^T]^T \in \mathbb{R}^6$. By maintaining the full $6 \times 6$ covariance matrix $P$, the filter accurately captures and updates the cross-correlations between positional errors and velocity estimates. 

This pure kinematic observer serves as a foundational baseline before introducing more complex rigid body dynamics, hydrodynamic forces, or transitioning to an Extended Kalman Filter (EKF) for nonlinear system tracking.

## Key Features
* **Coupled Matrix Formulation:** Implements the complete discrete-time algebraic Riccati equations for prediction and correction, avoiding dimensionality shortcuts.
* **High-Performance Linear Algebra:** Leverages the industry-standard `Eigen3` C++ library for computationally efficient, mathematically expressive matrix operations.
* **Stochastic Simulation Environment:** Includes a deterministic true-trajectory generator injected with Gaussian white noise ($v \sim \mathcal{N}(0, R)$) to validate filter convergence, transient response, and noise rejection capabilities in real-time.

## Dependencies
This project requires the **Eigen** linear algebra library for C++. 

**For macOS users:**
Install Eigen easily via Homebrew:
```bash
brew install eigen