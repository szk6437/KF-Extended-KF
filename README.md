# Kalman Filter Implementations (C++ / Eigen)

This repository contains a collection of robust C++ implementations for state estimation, progressing from standard linear Kalman Filters to correlated noise models, Extended Kalman Filters (EKFs), and Unscented Kalman Filters (UKFs). All algorithms utilize the `Eigen3` linear algebra library for computationally efficient, full-matrix formulations.

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

## 3. Colored Process Noise (State Augmentation)
**File:** `ColoredNoiseKF.cpp`

Standard filters assume process noise is purely white ($w_k \sim \mathcal{N}(0, Q)$). When environmental disturbances exhibit "memory" or trends over time (colored noise), standard filters can lag or diverge.

* **Methodology:** Implements **State Augmentation**. The colored noise is modeled as a first-order autoregressive AR(1) process: 
  $$w_k = A_w w_{k-1} + \xi_{k-1}$$
* **State Vector:** The $3 \times 1$ primary state and the $3 \times 1$ noise state are fused into a comprehensive $6 \times 1$ augmented state vector $X = [x^T, w^T]^T$.
* **Augmented Dynamics:**
  $$X_k = \begin{bmatrix} F & I \\ 0 & A_w \end{bmatrix} X_{k-1} + \begin{bmatrix} 0 \\ I \end{bmatrix} \xi_{k-1}$$
By treating the colored disturbance as a trackable state, the filter actively estimates and rejects the autocorrelated environmental noise in real-time.

## 4. Extended Kalman Filter (Nonlinear Bilinear System)
**File:** `ExtendedKalmanFilter.cpp`

Unlike the linear variants, this filter handles systems where the transition model and observation model are strictly non-linear functions ($f$ and $h$). It calculates the Jacobian matrices at every time step to artificially linearize the equations around the current working point using multivariate Taylor series expansions.

* **System Model:** A 2D bilinear system where the scalar control input ($u$) couples directly into the state transition matrix ($x_k u_k$). 
* **Observation Model:** Simulates a nonlinear Range and Bearing sensor:
  $$z_k = \begin{bmatrix} \sqrt{x_{1,k}^2 + x_{2,k}^2} \\ \arctan(x_{2,k} / x_{1,k}) \end{bmatrix} + v_k$$
* **Methodology:** Analytically computes the dynamic Jacobians $F_k = \frac{\partial f}{\partial x}$ and $H_k = \frac{\partial h}{\partial x}$ at each time step to propagate the Riccati equations, incorporating angular wrap-around handling for the bearing innovation.

## 5. Unscented Kalman Filter (CTRV Model)
**File:** `UnscentedKF.cpp`

Unlike the EKF, which relies on first-order linear approximations (Jacobians) that can diverge under severe non-linearities, this implementation utilizes the Unscented Transform (UT). It achieves 3rd-order Taylor series accuracy for Gaussian inputs without requiring analytical derivatives.

* **State Vector:** Tracks a maneuvering target using a Constant Turn Rate and Velocity (CTRV) kinematic model: $x = [p_x, p_y, v, \psi, \dot{\psi}]^T$.
* **Process Model ($f$):** Evaluates non-linear state transitions $x_{k+1} = f(x_k, \Delta t)$. 
  If $\dot{\psi} \neq 0$:
  $$p_x^{(k+1)} = p_x^{(k)} + \frac{v}{\dot{\psi}} \left(\sin(\psi + \dot{\psi}\Delta t) - \sin(\psi)\right)$$
  $$p_y^{(k+1)} = p_y^{(k)} + \frac{v}{\dot{\psi}} \left(-\cos(\psi + \dot{\psi}\Delta t) + \cos(\psi)\right)$$
  If $\dot{\psi} = 0$ (Straight line approximation):
  $$p_x^{(k+1)} = p_x^{(k)} + v \cos(\psi) \Delta t$$
  $$p_y^{(k+1)} = p_y^{(k)} + v \sin(\psi) \Delta t$$
  Velocity and yaw updates for both conditions:
  $$v^{(k+1)} = v^{(k)}$$
  $$\psi^{(k+1)} = \psi^{(k)} + \dot{\psi}\Delta t$$
  $$\dot{\psi}^{(k+1)} = \dot{\psi}^{(k)}$$
* **Observation Model ($h$):** Fuses polar radar measurements consisting of range and bearing $z = [r, \phi]^T$:
  $$r = \sqrt{p_x^2 + p_y^2}$$
  $$\phi = \arctan(p_y / p_x)$$
* **Methodology:** Deterministically extracts $2n+1$ sigma points via Cholesky decomposition of the scaled state covariance matrix $P$. These points are propagated directly through the non-linear process and measurement equations. The transformed points are then recombined using specific weights to capture the posterior mean and covariance dynamically.

---

## Dependencies & Compilation (macOS)
This project requires the **Eigen3** linear algebra library for all matrix operations. 

Compile the individual experiments using `clang++`, ensuring the compiler is linked to the Eigen include directory:

```bash
# Compile the 6-DOF Tracker
clang++ -std=c++17 -I /opt/homebrew/include/eigen3 KalmanFilter.cpp -o kf_tracker

# Compile the Correlated Noise Experiment
clang++ -std=c++17 -I /opt/homebrew/include/eigen3 CorrelatedNoiseKF.cpp -o ckf_tracker

# Compile the Colored Noise Experiment
clang++ -std=c++17 -I /opt/homebrew/include/eigen3 ColoredNoiseKF.cpp -o colored_tracker

# Compile the Extended Kalman Filter
clang++ -std=c++17 -I /opt/homebrew/include/eigen3 ExtendedKalmanFilter.cpp -o extended_kf

# Compile the Unscented Kalman Filter
clang++ -std=c++17 -I /opt/homebrew/include/eigen3 UnscentedKF.cpp -o unscented_kf
