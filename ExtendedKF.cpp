/**
 * @file ExtendedKalmanFilter.cpp
 * @brief Extended Kalman Filter (EKF) for a 2D Bilinear System with Range/Bearing Sensors.
 * * Designed for robust tracking of a nonlinear system where control inputs couple
 * with the states. Uses analytic Jacobians (Taylor series linearization) to 
 * compute real-time transition and measurement matrices.
 * * Dependencies: Eigen3 (https://eigen.tuxfamily.org/)
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

class ExtendedKalmanFilter {
private:
    Vector2d x; // State estimate [x1, x2]^T
    Matrix2d P; // Covariance estimate
    Matrix2d Q; // Process noise covariance
    Matrix2d R; // Measurement noise covariance

public:
    ExtendedKalmanFilter(const Vector2d& x0, const Matrix2d& P0, 
                         const Matrix2d& Q_in, const Matrix2d& R_in) 
        : x(x0), P(P0), Q(Q_in), R(R_in) {}

    void predict(double u) {
        // 1. Calculate the dynamic State Transition Jacobian (F)
        Matrix2d F;
        F(0, 0) = 1.0 + 0.001 * u;
        F(0, 1) = 0.01;
        F(1, 0) = 0.01;
        F(1, 1) = 1.0 - 0.004 * u;

        // 2. Predict State using nonlinear function f(x, u)
        Vector2d x_pred;
        x_pred(0) = x(0) + 0.01 * x(1) + 0.009 * u + 0.001 * x(0) * u;
        x_pred(1) = 0.01 * x(0) + x(1) + 0.009 * u - 0.004 * x(1) * u;
        x = x_pred;

        // 3. Predict Covariance
        P = F * P * F.transpose() + Q;
    }

    void update(const Vector2d& z) {
        // Safe-guard against division by zero at origin
        double range_sq = x(0)*x(0) + x(1)*x(1);
        if (range_sq < 1e-6) range_sq = 1e-6; 
        double range = std::sqrt(range_sq);

        // 1. Calculate Measurement Jacobian (H)
        Matrix2d H;
        H(0, 0) = x(0) / range;
        H(0, 1) = x(1) / range;
        H(1, 0) = -x(1) / range_sq;
        H(1, 1) = x(0) / range_sq;

        // 2. Predict Measurement h(x)
        Vector2d z_pred;
        z_pred(0) = range;
        z_pred(1) = std::atan2(x(1), x(0));

        // 3. Calculate Innovation (with angle wrapping correction)
        Vector2d y = z - z_pred;
        while (y(1) > M_PI)  y(1) -= 2.0 * M_PI;
        while (y(1) < -M_PI) y(1) += 2.0 * M_PI;

        // 4. Standard Kalman Gain and Update
        Matrix2d S = H * P * H.transpose() + R;
        Matrix2d K = P * H.transpose() * S.inverse();

        x = x + K * y;
        Matrix2d I = Matrix2d::Identity();
        P = (I - K * H) * P;
    }

    Vector2d getState() const { return x; }
};

int main() {
    std::cout << "--- Extended Kalman Filter: Nonlinear Bilinear System ---\n\n";

    // 1. Initialize Filter Parameters
    Matrix2d Q = Matrix2d::Identity() * 0.01; // Process noise
    Matrix2d R;                               // Measurement noise (Range, Bearing)
    R << 0.1, 0.0, 
         0.0, 0.05; 

    Vector2d x_init(0.5, 0.5); // Start slightly off origin to avoid singularity
    Matrix2d P_init = Matrix2d::Identity() * 1.0;

    ExtendedKalmanFilter ekf(x_init, P_init, Q, R);

    // 2. Setup Simulation Noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> w_dist(0.0, 0.1); // Process noise (3-sigma = 0.3, respects bound)
    std::normal_distribution<> v_range(0.0, std::sqrt(0.1));
    std::normal_distribution<> v_bearing(0.0, std::sqrt(0.05));

    Vector2d true_x(0.5, 0.5);
    int steps = 500; // Rigorous validation length
    double rmse_sum = 0.0;

    std::cout << std::fixed << std::setprecision(3);
    for (int k = 1; k <= steps; ++k) {
        // A. Define control input (oscillating bounded input |u| <= 2)
        double u = 1.5 * std::sin(k * 0.2);

        // B. Progress True System f(x, u) + w
        Vector2d true_next;
        true_next(0) = true_x(0) + 0.01*true_x(1) + 0.009*u + 0.001*true_x(0)*u + w_dist(gen);
        true_next(1) = 0.01*true_x(0) + true_x(1) + 0.009*u - 0.004*true_x(1)*u + w_dist(gen);
        true_x = true_next;

        // C. Generate Nonlinear Measurement h(x) + v
        double true_range = std::sqrt(true_x(0)*true_x(0) + true_x(1)*true_x(1));
        double true_bearing = std::atan2(true_x(1), true_x(0));
        Vector2d z(true_range + v_range(gen), true_bearing + v_bearing(gen));

        // D. EKF Execution
        ekf.predict(u);
        ekf.update(z);
        Vector2d est_x = ekf.getState();

        // E. Calculate Error
        double error_norm = (true_x - est_x).norm();
        rmse_sum += error_norm * error_norm;

        // F. Print Output (Scaled for 1000 iterations)
        if (k % 100 == 0 || k == 1) {
            std::cout << "Step " << std::setw(4) << k 
                      << " | Input u: " << std::setw(5) << u
                      << " | True X: [" << std::setw(6) << true_x(0) << ", " << std::setw(6) << true_x(1) << "]"
                      << " | Est X: ["  << std::setw(6) << est_x(0)  << ", " << std::setw(6) << est_x(1)  << "]\n";
        }
    }

    std::cout << "\nFinal RMSE: " << std::sqrt(rmse_sum / steps) << "\n";
    return 0;
}