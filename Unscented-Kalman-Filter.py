import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

class UKF:
    def __init__(self, dim_x, dim_z, alpha, beta, kappa, fx, hx):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.fx = fx
        self.hx = hx
        
        # Scaling parameters
        self.lam = alpha**2 * (dim_x + kappa) - dim_x
        self.c = dim_x + self.lam
        
        # Weights
        self.Wm = np.full(2 * dim_x + 1, 1.0 / (2.0 * self.c))
        self.Wc = np.full(2 * dim_x + 1, 1.0 / (2.0 * self.c))
        self.Wm[0] = self.lam / self.c
        self.Wc[0] = self.lam / self.c + (1.0 - alpha**2 + beta)
        
        # Matrices
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def generate_sigma_points(self, x, P):
        sigmas = np.zeros((2 * self.dim_x + 1, self.dim_x))
        # Add a small epsilon to the diagonal to ensure positive semi-definiteness
        U = linalg.cholesky(self.c * P + np.eye(self.dim_x) * 1e-8) 
        sigmas[0] = x
        for k in range(self.dim_x):
            sigmas[k + 1] = x + U[k]
            sigmas[self.dim_x + k + 1] = x - U[k]
        return sigmas

    def predict(self, dt):
        sigmas = self.generate_sigma_points(self.x, self.P)
        
        self.sigmas_f = np.zeros((2 * self.dim_x + 1, self.dim_x))
        for i in range(2 * self.dim_x + 1):
            self.sigmas_f[i] = self.fx(sigmas[i], dt)
            
        self.x = np.dot(self.Wm, self.sigmas_f)
        self.x[3] = (self.x[3] + np.pi) % (2 * np.pi) - np.pi # Wrap yaw
        
        self.P = self.Q.copy()
        for i in range(2 * self.dim_x + 1):
            y = self.sigmas_f[i] - self.x
            y[3] = (y[3] + np.pi) % (2 * np.pi) - np.pi
            self.P += self.Wc[i] * np.outer(y, y)

    def update(self, z):
        sigmas_h = np.zeros((2 * self.dim_x + 1, self.dim_z))
        for i in range(2 * self.dim_x + 1):
            sigmas_h[i] = self.hx(self.sigmas_f[i])
            
        zp = np.dot(self.Wm, sigmas_h)
        zp[1] = (zp[1] + np.pi) % (2 * np.pi) - np.pi
        
        S = self.R.copy()
        T = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(2 * self.dim_x + 1):
            y = sigmas_h[i] - zp
            y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi 
            
            S += self.Wc[i] * np.outer(y, y)
            
            x_diff = self.sigmas_f[i] - self.x
            x_diff[3] = (x_diff[3] + np.pi) % (2 * np.pi) - np.pi
            T += self.Wc[i] * np.outer(x_diff, y)
            
        K = np.dot(T, np.linalg.inv(S))
        
        y = z - zp
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        
        self.x = self.x + np.dot(K, y)
        self.x[3] = (self.x[3] + np.pi) % (2 * np.pi) - np.pi
        
        self.P = self.P - np.dot(K, np.dot(S, K.T))

# --- Simulation Specific Functions ---

def fx(state, dt):
    """ Constant Turn Rate and Velocity (CTRV) Model """
    px, py, v, yaw, yaw_rate = state
    
    if abs(yaw_rate) < 1e-5:
        px_p = px + v * dt * np.cos(yaw)
        py_p = py + v * dt * np.sin(yaw)
    else:
        px_p = px + (v / yaw_rate) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
        py_p = py + (v / yaw_rate) * (-np.cos(yaw + yaw_rate * dt) + np.cos(yaw))
        
    v_p = v
    yaw_p = yaw + yaw_rate * dt
    yaw_rate_p = yaw_rate
    
    return np.array([px_p, py_p, v_p, yaw_p, yaw_rate_p])

def hx(state):
    """ Radar Measurement Model: [Range, Bearing] """
    px, py = state[0], state[1]
    r = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    return np.array([r, phi])

def run_simulation():
    dt = 0.1
    steps = 300
    
    # Initialize UKF (dim_x=5 for CTRV)
    ukf = UKF(dim_x=5, dim_z=2, alpha=1e-3, beta=2, kappa=0, fx=fx, hx=hx)
    ukf.x = np.array([0., 0., 10., 0., 0.]) # [px, py, v, yaw, yaw_rate]
    ukf.P = np.diag([1.0, 1.0, 2.0, 0.5, 0.5])
    # Process noise: assume variation in longitudinal acceleration and yaw acceleration
    ukf.Q = np.diag([0.1, 0.1, 0.5, 0.1, 0.1]) 
    ukf.R = np.diag([1.0**2, 0.05**2]) # Measurement noise (Range, Bearing)
    
    # Ground Truth Initial State
    true_x = np.array([0., 0., 10., 0., 0.])
    
    history_true = []
    history_est = []
    history_meas = []
    
    for step in range(steps):
        # Inject a time-varying yaw rate to create a complex reference trajectory (S-Curve / Maneuvers)
        true_x[4] = 0.4 * np.sin(step * dt * 0.3) 
        
        # Update true state
        true_x = fx(true_x, dt)
        history_true.append(true_x.copy())
        
        # Generate noisy radar measurement
        r_true = np.sqrt(true_x[0]**2 + true_x[1]**2)
        phi_true = np.arctan2(true_x[1], true_x[0])
        z = np.array([r_true + np.random.randn() * 1.0, 
                      phi_true + np.random.randn() * 0.05])
        
        # Convert polar meas back to cartesian for plotting
        meas_x = z[0] * np.cos(z[1])
        meas_y = z[0] * np.sin(z[1])
        history_meas.append((meas_x, meas_y))
        
        # UKF Step
        ukf.predict(dt)
        ukf.update(z)
        history_est.append(ukf.x.copy())
        
    # Formatting outputs
    history_true = np.array(history_true)
    history_est = np.array(history_est)
    history_meas = np.array(history_meas)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.plot(history_true[:, 0], history_true[:, 1], 'k--', label='True Maneuvering Trajectory', linewidth=2)
    plt.scatter(history_meas[:, 0], history_meas[:, 1], c='r', marker='x', s=15, label='Radar Measurements', alpha=0.5)
    plt.plot(history_est[:, 0], history_est[:, 1], 'b-', label='UKF Estimate (CTRV)', linewidth=2)
    
    plt.title('UKF Tracking (CTRV Model)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()