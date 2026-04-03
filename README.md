# Kalman Filter Implementations (C++ / Eigen)

This repository contains a collection of robust C++ implementations for state estimation, progressing from standard linear Kalman Filters to correlated noise models, and eventually Extended Kalman Filters (EKFs). All algorithms utilize the `Eigen3` linear algebra library for computationally efficient, full-matrix formulations.

---

## 1. 6-DOF Kinematic State Estimation
**File:** `KalmanFilter.cpp`

A fully coupled 6-dimensional linear Kalman Filter designed for the kinematic state estimation of marine vehicles (e.g., AUVs). 

* **State Vector:** Tracks 3D position ($X, Y, Z$) and dynamically estimates hidden 3D velocity states (surge, sway, heave).
* **Methodology:** Avoids decoupled approximations by maintaining the full $6 \times 6$ covariance matrix ($P$) to accurately capture cross-correlations between positional innovations and velocity corrections. 
* **Model:** Constant Velocity (CV) kinematic tracking.

## 2. Correlated Process & Measurement Noise
**File:** `CorrelatedNoiseKF.cpp`

Standard Kalman Filter formulations assume process noise ($w$) and measurement noise ($v$) are completely independent. In severe environments, these noises often correlate ($E[w_k v_k^T] = C$).

* **State Vector:** A generalized $3 \times 3$ linear system.
* **Methodology:** Natively modifies the discrete-time algebraic Riccati equations to incorporate a known cross-covariance matrix ($C$). 
* **Validation:** The simulation constructs a $6 \times 6$ joint covariance matrix $\Sigma$ and uses Cholesky decomposition ($\Sigma = L L^T$) to accurately inject joint-normal multivariate noise into the simulation, proving the stability of the modified Kalman Gain and Innovation Covariance equations.

---

## Dependencies & Compilation (macOS)
This project requires the **Eigen** linear algebra library. 

Compile the individual experiments using `clang++`, ensuring the compiler is linked to the Eigen include directory:

```bash
# Compile the 6-DOF Tracker
clang++ -std=c++17 -I /opt/homebrew/include/eigen3 KalmanFilter.cpp -o kf_tracker

# Compile the Correlated Noise Experiment
clang++ -std=c++17 -I /opt/homebrew/include/eigen3 CorrelatedNoiseKF.cpp -o ckf_tracker