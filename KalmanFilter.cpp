/**
 * @file FullyCoupledKalmanFilter.cpp
 * @brief A fully coupled 6D Linear Kalman Filter for tracking 3D position and velocity.
 * * This implementation uses the Eigen library to handle the 6x6 matrix algebra required
 * for a Constant Velocity (CV) kinematic model. It simulates an object (e.g., an AUV) 
 * moving through 3D space, generates noisy sensor measurements, and filters them 
 * to estimate both the true position and the hidden velocity states.
 * * Dependencies: Eigen3 (https://eigen.tuxfamily.org/)
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <Eigen/Dense> // Standard C++ linear algebra library

using namespace Eigen;

class FullyCoupledKalmanFilter {
private:
    VectorXd x; // State vector (6x1): [x, y, z, vx, vy, vz]^T
    MatrixXd P; // Covariance matrix (6x6): Uncertainty in the state estimate
    MatrixXd Q; // Process noise matrix (6x6): Uncertainty in the kinematic model
    MatrixXd R; // Measurement noise matrix (3x3): Uncertainty in the sensor readings
    MatrixXd H; // Measurement mapping matrix (3x6): Maps the 6D state to a 3D measurement

public:
    /**
     * @brief Constructor to initialize the filter matrices.
     * @param q_pos Process noise variance for position.
     * @param q_vel Process noise variance for velocity.
     * @param r_noise Measurement noise variance (sensor inaccuracy).
     * @param p_init Initial uncertainty variance for the state estimate.
     */
    FullyCoupledKalmanFilter(double q_pos, double q_vel, double r_noise, double p_init) {
        // Initialize state to origin [0, 0, 0, 0, 0, 0]^T
        x = VectorXd::Zero(6); 
        
        // Initialize uncertainty matrix
        P = MatrixXd::Identity(6, 6) * p_init;
        
        // Construct the Process Noise Matrix (Q)
        Q = MatrixXd::Zero(6, 6);
        Q.block(0, 0, 3, 3) = MatrixXd::Identity(3, 3) * q_pos; 
        Q.block(3, 3, 3, 3) = MatrixXd::Identity(3, 3) * q_vel; 
        
        // Construct the Measurement Noise Matrix (R)
        R = MatrixXd::Identity(3, 3) * r_noise;
        
        // Construct the Measurement Mapping Matrix (H)
        // H extracts the first 3 elements (position) from the 6D state vector
        H = MatrixXd::Zero(3, 6);
        H.block(0, 0, 3, 3) = MatrixXd::Identity(3, 3); 
    }

    /**
     * @brief The Prediction Step (Time Update).
     * Advances the internal kinematic model by 'dt' seconds.
     * @param dt The time step since the last update.
     */
    void predict(double dt) {
        // 1. Construct the State Transition Matrix (F)
        // This applies the basic kinematic equation: position_new = position_old + (velocity * dt)
        MatrixXd F = MatrixXd::Identity(6, 6);
        F(0, 3) = dt; // x += vx * dt
        F(1, 4) = dt; // y += vy * dt
        F(2, 5) = dt; // z += vz * dt

        // 2. Predict State (x = F * x)
        x = F * x;

        // 3. Predict Covariance (P = F * P * F^T + Q)
        P = F * P * F.transpose() + Q;
    }

    /**
     * @brief The Correction Step (Measurement Update).
     * Corrects the predicted state using a new sensor reading.
     * @param z The 3D measurement vector [x, y, z]^T.
     */
    void update(Vector3d z) {
        // 1. Calculate the Innovation (Error between measurement and prediction)
        Vector3d y = z - (H * x);

        // 2. Calculate the Innovation Covariance (S) and Kalman Gain (K)
        MatrixXd S = H * P * H.transpose() + R;
        MatrixXd K = P * H.transpose() * S.inverse();

        // 3. Update the State Estimate
        x = x + (K * y);

        // 4. Update the Covariance Matrix
        MatrixXd I = MatrixXd::Identity(6, 6);
        P = (I - K * H) * P;
    }

    // --- Getters ---
    Vector3d getPosition() const { return x.head(3); }
    Vector3d getVelocity() const { return x.tail(3); }
};

int main() {
    // 1. Initialize the Filter
    // Process noise is kept low, measurement noise is kept high to demonstrate filtering
    FullyCoupledKalmanFilter kf(0.01, 0.05, 0.5, 1.0);

    // 2. Setup Noise Generator for the Simulation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, 0.7); // standard deviation ~0.7 yields ~0.5 variance

    std::cout << "--- Fully Coupled 6D Kalman Filter ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    // 3. True Simulation Kinematics
    double dt = 1.0; 
    Vector3d true_pos(0.0, 0.0, 0.0);
    Vector3d true_vel(1.0, 0.2, -0.1); // Constant surge, sway, and heave rates

    // 4. Main Simulation Loop
    for (int step = 1; step <= 50; ++step) {
        // A. Move the vehicle in the real world
        true_pos += true_vel * dt;

        // B. Generate a noisy sensor reading
        Vector3d z(
            true_pos.x() + noise(gen),
            true_pos.y() + noise(gen),
            true_pos.z() + noise(gen)
        );

        // C. The Kalman Filter Cycle
        kf.predict(dt); 
        kf.update(z);   

        // D. Extract the results
        Vector3d est_pos = kf.getPosition();
        
        // E. Print the output (True Position vs Estimated Position)
        std::cout << "Step " << std::setw(2) << step 
                  << " | True Pos: (" << std::setw(5) << true_pos.x() << ", " << std::setw(5) << true_pos.y() << ", " << std::setw(5) << true_pos.z() << ") "
                  << "| Est Pos:  (" << std::setw(5) << est_pos.x() << ", " << std::setw(5) << est_pos.y() << ", " << std::setw(5) << est_pos.z() << ")\n";
    }

    return 0;
}