/**
 * @file CorrelatedKalmanFilter.cpp
 * @brief 3D Linear Kalman Filter handling correlated process and measurement noises.
 * * This implementation natively modifies the Riccati equations to account for a 
 * known cross-covariance matrix (C) between the process noise and measurement noise.
 * It includes a Cholesky-decomposition-based simulation environment to accurately
 * generate the correlated joint-normal distributions for experimental validation.
 * * Dependencies: Eigen3 (https://eigen.tuxfamily.org/)
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Cholesky> // Required for generating correlated noises

using namespace Eigen;

class CorrelatedKalmanFilter {
private:
    Vector3d x; // State estimate
    Matrix3d P; // Estimate covariance
    Matrix3d F; // State transition matrix
    Matrix3d H; // Measurement matrix
    Matrix3d Q; // Process noise covariance
    Matrix3d R; // Measurement noise covariance
    Matrix3d C; // Cross-covariance between process and measurement noise

public:
    CorrelatedKalmanFilter(const Matrix3d& F_in, const Matrix3d& H_in, 
                           const Matrix3d& Q_in, const Matrix3d& R_in, const Matrix3d& C_in,
                           const Vector3d& x0, const Matrix3d& P0) 
        : F(F_in), H(H_in), Q(Q_in), R(R_in), C(C_in), x(x0), P(P0) {}

    void predict() {
        // Standard state and covariance prediction
        x = F * x;
        P = F * P * F.transpose() + Q;
    }

    void update(const Vector3d& z) {
        // 1. Calculate Innovation
        Vector3d y = z - (H * x);

        // 2. Innovation Covariance (S) with Cross-Covariance terms
        Matrix3d S = H * P * H.transpose() + R + H * C + C.transpose() * H.transpose();

        // 3. Kalman Gain (K) modified for correlated noise
        Matrix3d K = (P * H.transpose() + C) * S.inverse();

        // 4. Update State
        x = x + (K * y);

        // 5. Update Covariance (Symmetric form: P = P - K * S * K^T)
        P = P - K * S * K.transpose();
    }

    Vector3d getState() const { return x; }
};

int main() {
    std::cout << "--- 3x3 Correlated Noise Kalman Filter Experiment ---\n\n";

    // 1. Define the 3x3 Dummy Linear System
    Matrix3d F;
    F << 0.90, 0.05, 0.00,
         0.00, 0.95, 0.05,
         -0.01, 0.00, 0.90; // Slightly coupled, stable dynamics

    Matrix3d H = Matrix3d::Identity(); // Full state observability

    // 2. Define Covariances
    Matrix3d Q = Matrix3d::Identity() * 0.1;
    Matrix3d R = Matrix3d::Identity() * 0.5;
    
    // Define Cross-Covariance (C)
    // Assume process noise strongly correlates with measurement noise on the same axes
    Matrix3d C = Matrix3d::Identity() * 0.15; 

    // 3. Ensure the Joint Covariance Matrix is Positive Definite for Simulation
    MatrixXd Sigma(6, 6);
    Sigma << Q, C,
             C.transpose(), R;
             
    LLT<MatrixXd> lltOfSigma(Sigma);
    if (lltOfSigma.info() == Eigen::NumericalIssue) {
        std::cerr << "Error: Joint covariance matrix Sigma is not positive definite!\n";
        return -1;
    }
    MatrixXd L = lltOfSigma.matrixL(); // Cholesky lower triangle

    // 4. Initialize the Filter
    Vector3d x_est_init = Vector3d::Zero();
    Matrix3d P_init = Matrix3d::Identity() * 1.0;
    CorrelatedKalmanFilter kf(F, H, Q, R, C, x_est_init, P_init);

    // 5. Setup Random Number Generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> std_norm(0.0, 1.0);

    // 6. Simulation Loop
    Vector3d true_x = Vector3d::Ones(); // Start at [1, 1, 1]
    double rmse_sum = 0.0;
    int steps = 50;

    std::cout << std::fixed << std::setprecision(3);
    for (int k = 1; k <= steps; ++k) {
        // A. Generate Correlated Noise via Cholesky Decomposition
        VectorXd n_standard(6);
        for(int i=0; i<6; ++i) n_standard(i) = std_norm(gen);
        
        VectorXd correlated_noise = L * n_standard;
        Vector3d w = correlated_noise.head(3); // Process noise
        Vector3d v = correlated_noise.tail(3); // Measurement noise

        // B. Progress True System Dynamics
        true_x = F * true_x + w;

        // C. Generate Measurement
        Vector3d z = H * true_x + v;

        // D. Filter Execution
        kf.predict();
        kf.update(z);
        Vector3d est_x = kf.getState();

        // E. Calculate Error
        double error_norm = (true_x - est_x).norm();
        rmse_sum += error_norm * error_norm;

        if (k % 10 == 0 || k == 1) {
            std::cout << "Step " << std::setw(2) << k 
                      << " | True: [" << std::setw(6) << true_x(0) << ", " << std::setw(6) << true_x(1) << ", " << std::setw(6) << true_x(2) << "]"
                      << " | Est: ["  << std::setw(6) << est_x(0)  << ", " << std::setw(6) << est_x(1)  << ", " << std::setw(6) << est_x(2)  << "]"
                      << " | Error: " << error_norm << "\n";
        }
    }

    std::cout << "\nFinal Root Mean Square Error (RMSE): " << std::sqrt(rmse_sum / steps) << "\n";
    return 0;
}