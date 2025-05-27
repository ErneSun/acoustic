import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 0.01            # Pipe radius [m]
dr = 0.0001         # Spatial resolution [m]
inner_r = R - 0.002 # Detonation band inner radius
outer_r = R         # Detonation band outer radius
wave_pressure = 10  # Peak pressure [Pa]
sigma = 0.001       # Gaussian tail width [m]
omega = 1400 / R    # Angular speed [rad/s]
t = 0.0             # Initial time

# Physical parameters for Westervelt equation
rho0 = 1.225         # air density [kg/m^3]
c0 = 343.0           # sound speed [m/s]
beta = 1.2           # nonlinearity coefficient
delta = 2e-5         # dissipative term coefficient (approximate)
dt = 1e-9            # time step [s]
t_end = 1e-3         # end time [s]
n_steps = int(t_end / dt)

# Meshgrid
x = np.arange(-R, R + dr, dr)
y = np.arange(-R, R + dr, dr)
X, Y = np.meshgrid(x, y)

# Initial fields for Westervelt equation
p = np.zeros_like(X)
dpdt = np.zeros_like(X)
p_old = np.zeros_like(X)

# Source term function
def source_term(X, Y, t):
    theta = np.arctan2(Y, X) % (2*np.pi)
    r = np.sqrt(X**2 + Y**2)
    theta0 = (omega * t) % (2*np.pi)
    dtheta = (theta - theta0 + np.pi) % (2*np.pi) - np.pi
    mask_ring = (r >= inner_r) & (r <= outer_r) & (dtheta >= 0) & (dtheta*R < 3*sigma)
    S = np.zeros_like(X)+1
    # time_decay = np.exp(t / (5e-5))  # 可调节的时间尺度因子
    S[mask_ring] = wave_pressure * np.exp( (dtheta[mask_ring] * R)**2 / (2 * sigma**2)) #* time_decay
    return S

######################
# Time stepping: Westervelt equation (p, dpdt update)
for n in range(n_steps):
    t = n * dt
    S = source_term(X, Y, t)

    if abs(t % 1e-4) < dt:
        p_masked = np.ma.masked_where((X**2 + Y**2) > R**2, p)
        plt.figure(figsize=(6,5))
        plt.contourf(X*1000, Y*1000, p_masked, levels=100, cmap='inferno')
        plt.colorbar(label='Pressure (Pa)')
        plt.title(f"Pressure Field at t = {t*1e6:.1f} μs")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f"pressure_t_{int(t*1e6):05d}us.png", dpi=200)
        plt.close()
