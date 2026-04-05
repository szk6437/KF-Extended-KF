/**
 * @file ColoredNoiseKF.cpp
 * @brief Kalman Filter for systems with Colored (Auto-correlated) Process Noise.
 * * This implementation uses the State Augmentation method. It takes a 3D linear system
 * affected by colored process noise and augments the noise into the state vector, 
 * expanding the internal tracking to a 6D state-space model. 
 * * The colored noise is modeled as a discrete first-order autoregressive AR(1) process: 
 * w_k = A_w * w_{k-1} + \xi_{k-1}, where \xi is true white noise.
 * * Dependencies: Eigen3 (https://eigen.tuxfamily.org/)
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

class ColoredNoiseKF {
private:
    VectorXd X; // Augmented state vector (6x1): [x^T, w^T]^T
    MatrixXd P; // Augmented covariance matrix (6x6)
    MatrixXd F_aug; // Augmented state transition matrix (6x6)
    MatrixXd H_aug; // Augmented measurement matrix (3x6)
    MatrixXd Q_aug; // Augmented process noise covariance (6x6)
    Matrix3d R;     // Measurement noise covariance (3x3)

public:
    /**
     * @brief Constructs the Augmented Kalman Filter
     * @param F Original system transition matrix (3x3)
     * @param H Original measurement matrix (3x3)
     * @param A_w Colored noise transition matrix / AR(1) coefficients (3x3)
     * @param Q_xi Variance of the underlying driving white noise (3x3)
     * @param R_in Measurement noise covariance (3x3)
     */
    ColoredNoiseKF(const Matrix3d& F, const Matrix3d& H, 
                   const Matrix3d& A_w, const Matrix3d& Q_xi, const Matrix3d& R_in) {
        
        // 1. Initialize augmented state to zero
        X = VectorXd::Zero(6);
        P = MatrixXd::Identity(6, 6) * 1.0;
        R = R_in;

        // 2. Construct Augmented State Transition Matrix (F_aug)
        // [ F   I ]
        // [ 0  A_w]
        F_aug = MatrixXd::Zero(6, 6);
        F_aug.block(0, 0, 3, 3) = F;
        F_aug.block(0, 3, 3, 3) = Matrix3d::Identity();
        F_aug.block(3, 3, 3, 3) = A_w;

        // 3. Construct Augmented Measurement Matrix (H_aug)
        // [ H   0 ]
        H_aug = MatrixXd::Zero(3, 6);
        H_aug.block(0, 0, 3, 3) = H;

        // 4. Construct Augmented Process Noise Matrix (Q_aug)
        // [ 0    0  ]
        // [ 0  Q_xi ]
        Q_aug = MatrixXd::Zero(6, 6);
        Q_aug.block(3, 3, 3, 3) = Q_xi;
    }

    void predict() {
        X = F_aug * X;
        P = F_aug * P * F_aug.transpose() + Q_aug;
    }

    void update(const Vector3d& z) {
        Vector3d y = z - (H_aug * X); // Innovation
        Matrix3d S = H_aug * P * H_aug.transpose() + R; // Innovation covariance
        MatrixXd K = P * H_aug.transpose() * S.inverse(); // Kalman Gain (6x3)

        X = X + (K * y);
        MatrixXd I = MatrixXd::Identity(6, 6);
        P = (I - K * H_aug) * P;
    }

    // Returns just the primary 3D system state
    Vector3d getPrimaryState() const { return X.head(3); }
    
    // Returns the estimated colored noise currently acting on the system
    Vector3d getEstimatedNoise() const { return X.tail(3); } 
};

int main() {
    std::cout << "--- Colored Process Noise KF (State Augmentation) ---\n\n";

    // 1. Define the 3x3 Primary System
    Matrix3d F;
    F << 0.95, 0.05, 0.00,
         0.00, 0.90, 0.10,
         0.00, 0.00, 0.95; 

    Matrix3d H = Matrix3d::Identity(); 

    // 2. Define the Colored Noise Model (AR(1) Process)
    // A_w defines how strongly the noise correlates with its previous step
    Matrix3d A_w = Matrix3d::Identity() * 0.85; // High auto-correlation
    Matrix3d Q_xi = Matrix3d::Identity() * 0.05; // Driving white noise variance
    Matrix3d R = Matrix3d::Identity() * 0.5;     // Sensor noise

    ColoredNoiseKF kf(F, H, A_w, Q_xi, R);

    // 3. Setup Random Number Generation for Simulation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_dist(0.0, 1.0);

 // 4. Simulation Variables
    Vector3d true_x = Vector3d::Zero();
    Vector3d true_w = Vector3d::Zero(); // The actual colored noise in the environment
    
    double rmse_sum = 0.0;
    int steps = 1000; // <--- Increased from 50 to 1000 for a longer simulation

    std::cout << std::fixed << std::setprecision(3);
    for (int k = 1; k <= steps; ++k) {
        // A. Generate pure driving white noise (\xi)
        Vector3d xi(noise_dist(gen), noise_dist(gen), noise_dist(gen));
        xi = xi * std::sqrt(0.05); 

        // B. Advance the Colored Noise process in the environment
        true_w = A_w * true_w + xi;

        // C. Advance the True System State
        true_x = F * true_x + true_w;

        // D. Generate Noisy Measurement (v)
        Vector3d v(noise_dist(gen), noise_dist(gen), noise_dist(gen));
        v = v * std::sqrt(0.5);
        Vector3d z = H * true_x + v;

        // E. Filter Execution
        kf.predict();
        kf.update(z);

        Vector3d est_x = kf.getPrimaryState();

        // F. Calculate Error
        double error_norm = (true_x - est_x).norm();
        rmse_sum += error_norm * error_norm;

        // Print every 100 steps instead of every 10 to keep the terminal clean
        if (k % 100 == 0 || k == 1) {
            std::cout << "Step " << std::setw(4) << k 
                      << " | True X: [" << std::setw(6) << true_x(0) << ", " << std::setw(6) << true_x(1) << "]"
                      << " | Est X: ["  << std::setw(6) << est_x(0)  << ", " << std::setw(6) << est_x(1)  << "]"
                      << " | Error: " << error_norm << "\n";
        }
    }

    std::cout << "\nFinal Primary State RMSE: " << std::sqrt(rmse_sum / steps) << "\n";
    return 0;
}