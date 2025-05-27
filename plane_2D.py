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
t_rise = 1e-5        # pressure ramps up to full value in 10 µs
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


# -------------------------------------------------
# Read structured O‑grid from *.npz instead of Cartesian mesh
GRID_FILE = "O_grid_structured.npz"      # <- path to your mesh file
grid = np.load(GRID_FILE)

# Structured mesh in polar coordinates
X = grid["X"]            # shape (Nr, Ntheta)
Y = grid["Y"]
r = grid["r"]            # 1‑D radial coordinates, length Nr
theta = grid["theta"]    # 1‑D angular coordinates, length Ntheta
dr = r[1] - r[0]         # radial spacing
dtheta = theta[1] - theta[0]  # angular spacing
Nr, Ntheta = X.shape     # dimensions for later use

# Initial fields for Westervelt equation
p = np.zeros_like(X)
dpdt = np.zeros_like(X)
p_old = np.zeros_like(X)
p_older = p_old.copy()   # second‑previous pressure field for 2nd‑order time derivative


# Source term function
def source_term(X, Y, t):
    """
    Returns the desired pressure profile (Dirichlet boundary data)
    on the rotating detonation ring at time t. Outside the ring
    the array is zero.
    """
    theta = np.arctan2(Y, X) % (2 * np.pi)
    r_grid = np.sqrt(X ** 2 + Y ** 2)
    theta0 = (omega * t) % (2 * np.pi)
    dtheta = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi

    # Ring mask: between inner_r and outer_r, only on the "front" side of the wave
    mask_ring = (r_grid >= inner_r) & (r_grid <= outer_r) & (dtheta >= 0) & (dtheta * R < 3 * sigma)

    P = np.zeros_like(X)
    # Smooth ramp to avoid gigantic dp/dt at t = 0
    if t < t_rise:
        ramp = 0.5 * (1.0 - np.cos(np.pi * t / t_rise))   # 0→1 cosine window
    else:
        ramp = 1.0

    P[mask_ring] = ramp * wave_pressure * np.exp((dtheta[mask_ring] * R) ** 2 / (2 * sigma ** 2))
    return P

# Impose initial detonation pressure on the ring
P_init = source_term(X, Y, 0.0)
p[P_init > 0] = P_init[P_init > 0]

# -------------------------------------------------
# Helper: Laplacian in polar coordinates (reflective in r, periodic in theta)
def laplacian_polar(f):
    """
    2‑D Laplacian in polar coordinates with
    reflective (Neumann) boundaries at r = r_inner, r_outer
    and periodic boundary in θ.
    `f` has shape (Nr, Ntheta).
    """
    # Radial ghost cells by reflection
    f_pad = np.pad(f, ((1, 1), (0, 0)), mode="edge")   # reflect at both radial ends

    # Helpers
    f_r_plus  = f_pad[2:, :]     # f_{i+1,j}
    f_r_minus = f_pad[:-2, :]    # f_{i-1,j}
    f_th_plus  = np.roll(f, -1, axis=1)   # f_{i,j+1}
    f_th_minus = np.roll(f,  1, axis=1)   # f_{i,j-1}
    r_mat = r[:, None]            # broadcast to (Nr, Ntheta)

    # Polar‑coordinate Laplacian
    lap = (f_r_plus - 2.0 * f + f_r_minus) / dr**2 \
        + (1.0 / r_mat) * (f_r_plus - f_r_minus) / (2.0 * dr) \
        + (f_th_plus - 2.0 * f + f_th_minus) / (r_mat**2 * dtheta**2)

    return lap

######################
# Time stepping: Westervelt equation (p, dpdt update)
for n in range(n_steps):
    t = n * dt
    P_bc = source_term(X, Y, t)

    # Laplacian in polar coordinates with reflective/periodic BCs
    lap_p    = laplacian_polar(p)
    lap_dpdt = laplacian_polar(dpdt)

    # --- Stable nonlinear term: use finite‑difference second derivative, no /dt ---
    d2pdt2_fd = (p - 2.0 * p_old + p_older) / dt**2
    nonlinear_term = (beta / (rho0 * c0**2)) * (2.0 * p * dpdt**2 + p**2 * d2pdt2_fd)

    dp2dt2 = c0**2 * lap_p + nonlinear_term + delta * lap_dpdt

    # Verlet-like update
    p_new = 2*p - p_old + dt**2 * dp2dt2
    # Dirichlet boundary: keep the detonation ring at the prescribed pressure
    mask_bc = P_bc > 0
    p_new[mask_bc] = P_bc[mask_bc]
    dpdt[mask_bc] = 0.0
    dpdt = (p_new - p_old) / (2 * dt)
    p_older = p_old.copy()   # advance history: now p_old becomes p_older
    p_old = p.copy()
    p = p_new.copy()

    if n % 200 == 0:
        print(f"Step {n}/{n_steps}, t = {t:.2e}")
    # 每1e-4秒输出一次当前的压力场图像
    if abs(t % 1e-4) < dt:
        plt.figure(figsize=(6,5))
        plt.contourf(X*1000, Y*1000, p, levels=100, cmap='inferno')
        plt.colorbar(label='Pressure (Pa)')
        plt.title(f"Pressure Field at t = {t*1e6:.1f} μs")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f"pressure_t_{int(t*1e6):05d}us.png", dpi=200)
        plt.close()

# Evaluate and plot
S0 = p

plt.figure(figsize=(6,5))
plt.contourf(X*1000, Y*1000, S0, levels=100, cmap='inferno')
plt.colorbar(label='Source Pressure (Pa)')
plt.title("Initial Detonation Source Distribution (t = 0)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("source_term_initial.png", dpi=300)
plt.show()